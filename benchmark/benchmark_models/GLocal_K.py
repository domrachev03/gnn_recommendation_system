from time import time
from scipy.sparse import csc_matrix
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter

torch.manual_seed(1284)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data_100k(path='./', delimiter='\t'):

    train = np.loadtxt(path+'u1.base', skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(path+'u1.test', skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:,0]).size  # num of users
    n_m = np.unique(total[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_r[train[i,1]-1, train[i,0]-1] = train[i,2]

    for i in range(n_test):
        test_r[test[i,1]-1, test[i,0]-1] = test[i,2]

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m


def local_kernel(u, v):
    dist = torch.norm(u - v, p=2, dim=2)
    hat = torch.clamp(1. - dist**2, min=0.)
    return hat


class KernelLayer(nn.Module):
    def __init__(self, n_in, n_hid, n_dim, lambda_s, lambda_2, activation=nn.Sigmoid()):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_in, n_hid))
        self.u = nn.Parameter(torch.randn(n_in, 1, n_dim))
        self.v = nn.Parameter(torch.randn(1, n_hid, n_dim))
        self.b = nn.Parameter(torch.randn(n_hid))

        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2

        nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.b)
        self.activation = activation

    def forward(self, x):
        w_hat = local_kernel(self.u, self.v)
        
        sparse_reg = torch.nn.functional.mse_loss(w_hat, torch.zeros_like(w_hat))
        sparse_reg_term = self.lambda_s * sparse_reg
        
        l2_reg = torch.nn.functional.mse_loss(self.W, torch.zeros_like(self.W))
        l2_reg_term = self.lambda_2 * l2_reg

        W_eff = self.W * w_hat  # Local kernelised weight matrix
        y = torch.matmul(x, W_eff) + self.b
        y = self.activation(y)

        return y, sparse_reg_term + l2_reg_term


class KernelNet(nn.Module):
    def __init__(self, n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2))
            else:
                layers.append(KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2))
        layers.append(KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=nn.Identity()))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(0.33)

    def forward(self, x):
        total_reg = None
        for i, layer in enumerate(self.layers):
            x, reg = layer(x)
            if i < len(self.layers)-1:
                x = self.dropout(x)
            if total_reg is None:
                total_reg = reg
            else:
                total_reg += reg
        return x, total_reg


class CompleteNet(nn.Module):
    def __init__(self, kernel_net, n_u, n_m, n_hid, n_dim, n_layers, lambda_s, lambda_2, gk_size, dot_scale):
        super().__init__()
        self.gk_size = gk_size
        self.dot_scale = dot_scale
        self.local_kernel_net = kernel_net
        self.conv_kernel = torch.nn.Parameter(torch.randn(n_m, gk_size**2) * 0.1)
        nn.init.xavier_uniform_(self.conv_kernel, gain=torch.nn.init.calculate_gain("relu"))
      

    def forward(self, x, x_local):
        gk = self.global_kernel(x_local, self.gk_size, self.dot_scale)
        x = self.global_conv(x, gk)
        x, global_reg_loss = self.local_kernel_net(x)
        return x, global_reg_loss

    def global_kernel(self, input, gk_size, dot_scale):
        avg_pooling = torch.mean(input, dim=1)  # Item (axis=1) based average pooling
        avg_pooling = avg_pooling.view(1, -1)

        gk = torch.matmul(avg_pooling, self.conv_kernel) * dot_scale  # Scaled dot product
        gk = gk.view(1, 1, gk_size, gk_size)

        return gk

    def global_conv(self, input, W):
        input = input.unsqueeze(0).unsqueeze(0)
        conv2d = nn.LeakyReLU()(F.conv2d(input, W, stride=1, padding=1))
        return conv2d.squeeze(0).squeeze(0)


class Loss(nn.Module):
    def forward(self, pred_p, reg_loss, train_m, train_r):
        # L2 loss
        diff = train_m * (train_r - pred_p)
        sqE = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
        loss_p = sqE + reg_loss
        return loss_p


# Metrics
def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg


def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm


def call_ndcg(y_hat, y):
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num


# Train & Val
def train(data_path='data/raw/ml-100k/'):
    n_m, n_u, train_r, train_m, test_r, test_m = load_data_100k(path=data_path, delimiter='\t')
    # Common hyperparameter settings
    n_hid = 500         # size of hidden layers
    n_dim = 5           # inner AE embedding size
    n_layers = 2        # number of hidden layers
    gk_size = 3         # width=height of kernel for convolution

    # Hyperparameters to tune for specific case
    max_epoch_p = 500   # max number of epochs for pretraining
    max_epoch_f = 1000  # max number of epochs for finetuning
    patience_p = 5      # number of consecutive rounds of early stopping condition before actual stop for pretraining
    patience_f = 10     # and finetuning
    tol_p = 1e-4        # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
    tol_f = 1e-5        # and finetuning
    lambda_2 = 20.      # regularisation of number or parameters
    lambda_s = 0.006    # regularisation of sparsity of the final matrix
    dot_scale = 1       # dot product weight for global kernel
    
    model = KernelNet(n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2).double().to(device)
    complete_model = CompleteNet(
        model, n_u, n_m, n_hid, n_dim, n_layers, lambda_s, lambda_2, gk_size, dot_scale
    ).double().to(device)

    best_rmse_ep, best_mae_ep, best_ndcg_ep = 0, 0, 0
    best_rmse, best_mae, best_ndcg = float("inf"), float("inf"), 0

    time_cumulative = 0
    tic = time()

    # Pre-Training
    optimizer = torch.optim.AdamW(complete_model.local_kernel_net.parameters(), lr=0.001)

    def closure():
        optimizer.zero_grad()
        x = torch.Tensor(train_r).double().to(device)
        m = torch.Tensor(train_m).double().to(device)
        complete_model.local_kernel_net.train()
        pred, reg = complete_model.local_kernel_net(x)
        loss = Loss().to(device)(pred, reg, m, x)
        loss.backward()
        return loss

    last_rmse = np.inf
    counter = 0

    for i in range(max_epoch_p):
        optimizer.step(closure)
        complete_model.local_kernel_net.eval()
        t = time() - tic
        time_cumulative += t

        pre, _ = model(torch.Tensor(train_r).double().to(device))

        pre = pre.float().cpu().detach().numpy()

        error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
        test_rmse = np.sqrt(error)

        error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
        train_rmse = np.sqrt(error_train)

        if last_rmse-train_rmse < tol_p:
            counter += 1
        else:
            counter = 0

        last_rmse = train_rmse

        if patience_p == counter:
            print('.-^-._' * 12)
            print('PRE-TRAINING')
            print('Epoch:', i+1, 'test rmse:', test_rmse, 'train rmse:', train_rmse)
            print('Time:', t, 'seconds')
            print('Time cumulative:', time_cumulative, 'seconds')
            print('.-^-._' * 12)
            break

        if i % 50 != 0:
            continue
        print('.-^-._' * 12)
        print('PRE-TRAINING')
        print('Epoch:', i, 'test rmse:', test_rmse, 'train rmse:', train_rmse)
        print('Time:', t, 'seconds')
        print('Time cumulative:', time_cumulative, 'seconds')
        print('.-^-._' * 12)
        train_r_local = np.clip(pre, 1., 5.)

    train_r_local = np.clip(pre, 1., 5.)

    optimizer = torch.optim.AdamW(complete_model.parameters(), lr=0.001)

    def closure():
        optimizer.zero_grad()
        x = torch.Tensor(train_r).double().to(device)
        x_local = torch.Tensor(train_r_local).double().to(device)
        m = torch.Tensor(train_m).double().to(device)
        complete_model.train()
        pred, reg = complete_model(x, x_local)
        loss = Loss().to(device)(pred, reg, m, x)
        loss.backward()
        return loss

    last_rmse = np.inf
    counter = 0

    for i in range(max_epoch_f):
        optimizer.step(closure)
        complete_model.eval()
        t = time() - tic
        time_cumulative += t

        pre, _ = complete_model(torch.Tensor(train_r).double().to(device), torch.Tensor(train_r_local).double().to(device))

        pre = pre.float().cpu().detach().numpy()

        error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
        test_rmse = np.sqrt(error)

        error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
        train_rmse = np.sqrt(error_train)

        test_mae = (test_m * np.abs(np.clip(pre, 1., 5.) - test_r)).sum() / test_m.sum()
        train_mae = (train_m * np.abs(np.clip(pre, 1., 5.) - train_r)).sum() / train_m.sum()

        test_ndcg = call_ndcg(np.clip(pre, 1., 5.), test_r)
        train_ndcg = call_ndcg(np.clip(pre, 1., 5.), train_r)

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_rmse_ep = i+1

        if test_mae < best_mae:
            best_mae = test_mae
            best_mae_ep = i+1

        if best_ndcg < test_ndcg:
            best_ndcg = test_ndcg
            best_ndcg_ep = i+1

        if last_rmse-train_rmse < tol_f:
            counter += 1
        else:
            counter = 0

        last_rmse = train_rmse

        if patience_f == counter:
            print('.-^-._' * 12)
            print('FINE-TUNING')
            print('Epoch:', i+1, 'test rmse:', test_rmse, 'test mae:', test_mae, 'test ndcg:', test_ndcg)
            print('Epoch:', i+1, 'train rmse:', train_rmse, 'train mae:', train_mae, 'train ndcg:', train_ndcg)
            print('Time:', t, 'seconds')
            print('Time cumulative:', time_cumulative, 'seconds')
            print('.-^-._' * 12)
            break

        if i % 50 != 0:
            continue

        print('.-^-._' * 12)
        print('FINE-TUNING')
        print('Epoch:', i, 'test rmse:', test_rmse, 'test mae:', test_mae, 'test ndcg:', test_ndcg)
        print('Epoch:', i, 'train rmse:', train_rmse, 'train mae:', train_mae, 'train ndcg:', train_ndcg)
        print('Time:', t, 'seconds')
        print('Time cumulative:', time_cumulative, 'seconds')
        print('.-^-._' * 12)

        # Final result
        print('Epoch:', best_rmse_ep, ' best rmse:', best_rmse)
        print('Epoch:', best_mae_ep, ' best mae:', best_mae)
        print('Epoch:', best_ndcg_ep, ' best ndcg:', best_ndcg)


def test():
    pass


if __name__ == '__main__':
    train()
