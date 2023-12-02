from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import argparse

sys.path.append('benchmark')
sys.path.append('src/data')

from metrics import get_metrics, get_metrics_names
from load_dataset import load_data_100k_np

torch.manual_seed(1284)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class KernelLayer(nn.Module):
    ''' Kernel layer of the model'''
    def __init__(self, n_in, n_hid, n_dim, lambda_s, lambda_2, activation=nn.Sigmoid()):
        super().__init__()
        # Initializing the parameters of the kernel
        self.W = nn.Parameter(torch.randn(n_in, n_hid))
        self.u = nn.Parameter(torch.randn(n_in, 1, n_dim))
        self.v = nn.Parameter(torch.randn(1, n_hid, n_dim))
        self.b = nn.Parameter(torch.randn(n_hid))

        # Regularization terms
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2

        # Weights initialization
        nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.b)
        self.activation = activation

    def local_kernel(self, u, v):
        ''' Local kernel, which is radial basis function'''
        dist = torch.norm(u - v, p=2, dim=2)
        hat = torch.clamp(1. - dist**2, min=0.)
        return hat

    def forward(self, x):
        w_hat = self.local_kernel(self.u, self.v)

        # Mean Squared w_hat
        sparse_reg = torch.nn.functional.mse_loss(w_hat, torch.zeros_like(w_hat))
        sparse_reg_term = self.lambda_s * sparse_reg

        # Mean Squared W
        l2_reg = torch.nn.functional.mse_loss(self.W, torch.zeros_like(self.W))
        l2_reg_term = self.lambda_2 * l2_reg

        # Local reparametrized weight matrix
        W_eff = self.W * w_hat  
        y = torch.matmul(x, W_eff) + self.b
        y = self.activation(y)

        return y, sparse_reg_term + l2_reg_term


class KernelNet(nn.Module):
    ''' Autoencoder networkm '''
    def __init__(self, n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2):
        super().__init__()

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2))
            else:
                # Kernels of the same size
                layers.append(KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2))

        # Last layer -- kernel without activation and with specified output dimension
        layers.append(KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=nn.Identity()))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(0.33)

    def forward(self, x):
        total_reg = None
        for i, layer in enumerate(self.layers):
            x, reg = layer(x)

            # Apply dropout on the last layer
            if i < len(self.layers)-1:
                x = self.dropout(x)

            # Regularization terms summation
            if total_reg is None:
                total_reg = reg
            else:
                total_reg += reg

        return x, total_reg


class GLocal_K(nn.Module):
    '''The GLocal-K network, finetuning on the global kernel '''
    def __init__(self, kernel_net, n_m, gk_size, dot_scale):
        super().__init__()

        # Size of the global kernel
        self.gk_size = gk_size
        # Pretrained KernelNet
        self.local_kernel_net = kernel_net
        self.dot_scale = dot_scale

        # Convolution layer init
        self.conv_kernel = torch.nn.Parameter(torch.randn(n_m, gk_size**2) * 0.1)
        nn.init.xavier_uniform_(self.conv_kernel, gain=torch.nn.init.calculate_gain("relu"))

    def global_kernel(self, input, gk_size, dot_scale):
        avg_pooling = torch.mean(input, dim=1)  # Item (axis=1) based average pooling
        avg_pooling = avg_pooling.view(1, -1)

        global_kernel = torch.matmul(avg_pooling, self.conv_kernel) * dot_scale  # Scaled dot product
        global_kernel = global_kernel.view(1, 1, gk_size, gk_size)

        return global_kernel

    def global_conv(self, input, W):
        input = input.unsqueeze(0).unsqueeze(0)
        conv2d = nn.LeakyReLU()(F.conv2d(input, W, stride=1, padding=1))
        return conv2d.squeeze(0).squeeze(0)

    def forward(self, x, x_local=None):
        if x_local is None:
            x_local, _ = self.local_kernel_net(x)
        # First, apply global kernel
        gk = self.global_kernel(x_local, self.gk_size, self.dot_scale)
        x = self.global_conv(x, gk)
        # Then run the Kenel Network
        x, global_reg_loss = self.local_kernel_net(x)
        return x, global_reg_loss


class Loss(nn.Module):
    '''L2 loss class'''
    def forward(self, pred_p, reg_loss, train_mask, train_rating):
        # Calculating Mean Square Error
        # Considers only initially non-empty elements
        diff = train_mask * (train_rating - pred_p)
        mse = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))

        # Applying loss regularization
        loss_value = mse + reg_loss
        return loss_value


def print_info(i: int, train_metrics: dict, test_metrics: dict, t, time_cumulative: float, is_pretraining=True):
    print('-' * 60)
    if is_pretraining:
        print('PRE-TRAINING')
    else:
        print('FINETUNING')

    print('Epoch:', i+1)
    for metric_name in train_metrics.keys():
        print(f'    Metric {metric_name}: train -- {train_metrics[metric_name]}, test -- {test_metrics[metric_name]}')
    print('Time per epoch:', t, 'seconds')
    print('Total time:', time_cumulative, 'seconds')
    print('-' * 60)


# Train & Val
def train(data_path='data/raw/ml-100k/', weights_output_dir='.', verbose=False):
    n_m, n_u, train_rating, train_mask, test_rating, test_mask = load_data_100k_np(path=data_path, delimiter='\t')

    # Common hyperparameter settings
    n_hid = 500         # Size of hidden layers
    n_dim = 5           # Inner AE embedding size
    n_layers = 2        # Number of hidden layers
    gk_size = 3         # Width=height of kernel for convolution

    # Hyperparameters to tune for specific case
    max_epoch_p = 500   # Max number of epochs for pretraining
    max_epoch_f = 1000  # Max number of epochs for finetuning
    patience_p = 5      # No of rounds for early stopping during pretraining
    patience_f = 5     # No of rounds for early stopping during finetuning
    tol_p = 1e-4        # RMSE diff threshold for early stopping during pretraining
    tol_f = 1e-5        # RMSE diff threshold for early stopping during finetuning
    lambda_2 = 20.      # L2 regularization
    lambda_s = 0.006    # L1 regularization
    dot_scale = 1       # Dot product weight for global kernel

    autoencoder_net = KernelNet(n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2).double().to(device)

    glocal_k = GLocal_K(
        autoencoder_net, n_m, gk_size, dot_scale
    ).double().to(device)

    # Compute execution time
    time_cumulative = 0
    tic = time()

    # Pre-Training
    optimizer = torch.optim.AdamW(glocal_k.local_kernel_net.parameters(), lr=1e-3)

    last_rmse = np.inf
    counter = 0

    loss_fn = Loss().to(device)

    for i in range(max_epoch_p):
        # Training step
        optimizer.zero_grad()

        x = torch.Tensor(train_rating).double().to(device)
        masks = torch.Tensor(train_mask).double().to(device)

        # Forward pass for local network
        glocal_k.local_kernel_net.train()
        preds, regularization_value = glocal_k.local_kernel_net(x)

        loss = loss_fn(preds, regularization_value, masks, x)

        loss.backward()
        optimizer.step()

        # Metrics computation
        glocal_k.local_kernel_net.eval()
        t = time() - tic
        time_cumulative += t

        # Calculating global predictions
        preds = np.clip(preds.float().cpu().detach().numpy(), 1, 5)

        train_metrics = get_metrics(preds, train_rating, train_mask)
        test_metrics = get_metrics(preds, test_rating, test_mask)

        # Early Stopping Criterion update
        if last_rmse - train_metrics['rmse'] < tol_p:
            counter += 1
        else:
            counter = 0

        last_rmse = train_metrics['rmse']

        # Trigerring Early Stopping
        if patience_p == counter:
            print_info(i, train_metrics, test_metrics, t, time_cumulative)
            break

        if i % 50 == 0:
            print_info(i, train_metrics, test_metrics, t, time_cumulative)

    # Predictions after pretraining from autoencoder
    train_rating_local = np.clip(preds, 1., 5.)

    optimizer = torch.optim.AdamW(glocal_k.parameters(), lr=0.001)

    metrics_info = get_metrics_names()

    best_metric_epoch = {metric[0]: 0 for metric in metrics_info}
    best_metric_value = {metric[0]: 0 if metric[1] else float("inf") for metric in metrics_info}

    last_rmse = np.inf
    counter = 0

    for i in range(max_epoch_f):
        # Training step
        optimizer.zero_grad()

        x = torch.Tensor(train_rating).double().to(device)
        x_local = torch.Tensor(train_rating_local).double().to(device)
        mask = torch.Tensor(train_mask).double().to(device)
        # Global finetuning
        glocal_k.train()
        preds, reg = glocal_k(x, x_local)

        loss = loss_fn(preds, reg, mask, x)
        loss.backward()
        optimizer.step()

        glocal_k.eval()
        t = time() - tic
        time_cumulative += t

        preds = np.clip(preds.float().cpu().detach().numpy(), 1, 5)

        train_metrics = get_metrics(preds, train_rating, train_mask)
        test_metrics = get_metrics(preds, test_rating, test_mask)

        for metric_name, is_bigger_better in metrics_info:
            # Updates both maximum and minimum, based on metrics information
            if (is_bigger_better and test_metrics[metric_name] > best_metric_value[metric_name]) or \
               (not is_bigger_better and test_metrics[metric_name] < best_metric_value[metric_name]):
                torch.save(glocal_k, f"{weights_output_dir}/best_model_{metric_name}.pt")
                best_metric_value[metric_name] = test_metrics[metric_name]
                best_metric_epoch[metric_name] = i+1

        if last_rmse-train_metrics['rmse'] < tol_f:
            counter += 1
        else:
            counter = 0
        last_rmse = train_metrics['rmse']

        # Trigerring Early Stopping
        if patience_f == counter:
            print_info(i, train_metrics, test_metrics, t, time_cumulative, is_pretraining=False)
            break

        if i % 50 == 0:
            print_info(i, train_metrics, test_metrics, t, time_cumulative, is_pretraining=False)

    print('='*80)
    # Final result
    for metric_name in best_metric_value.keys():
        print(f'Metirc {metric_name}: epoch {best_metric_epoch[metric_name]}, value: {best_metric_value[metric_name]}')


def evaluate(data_path='data/raw/ml-100k/', weights='models/glocal_k/best_model_rmse.pt', verbose=False):
    n_m, n_u, train_rating, _, test_rating, test_mask = load_data_100k_np(path=data_path, delimiter='\t')

    glocal_k = torch.load(weights, map_location=torch.device(device))

    x = torch.Tensor(train_rating).double().to(device)

    # Global finetuning
    glocal_k.eval()

    t_begin = time()

    preds = glocal_k(x)[0].float().cpu().detach().numpy()
    dt = time() - t_begin

    metrics = get_metrics(preds, test_rating, test_mask)

    print('EVALUATION')
    print(f'Weights {weights.split(r"/")[-1]}')
    for metric_name, metric_value in metrics.items():
        print(f'    Metric {metric_name}: {metric_value}')
    print('Evaluation time:', dt, 'seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='GLocal_K benchmark network training',
        description='''The script to run training or inference of the model. 

        Note: both training and inference were validated on the MovieLense-100k dataset only.'''
    )

    parser.add_argument('-e', '--evaluate', action='store_true', help="Specify this argument to run the evaluation of the model")
    parser.add_argument('-p', '--path', type=str, default='./', help='Path to the data')
    parser.add_argument('-w', '--weights', type=str, help='Directory to load/save weights (depending on execution mode). Note: for training, the directory is expected, meanwhile for inference the specific file should be specified')
    parser.add_argument('-v', '--verbose', action='store_true', help="Specify this argument for verbose output")

    args = parser.parse_args()
    if not args.evaluate:
        path = args.path if args.path[-1] == '/' else args.path + '/'
        train(data_path=path, weights_output_dir=args.weights, verbose=args.verbose)
    else:
        path = args.path if args.path[-1] == '/' else args.path + '/'
        evaluate(data_path=path, weights=args.weights)
# def train(data_path='data/raw/ml-100k/', weights_output_dir='.', verbose=False):