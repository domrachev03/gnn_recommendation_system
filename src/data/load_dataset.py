import wget
import os
import zipfile
import argparse
import numpy as np


def download_dataset(
    url: str,
    output_path: str = './' 
):
    fname = wget.download(url, out=output_path)
    with zipfile.ZipFile(fname, 'r') as zip:
        zip.extractall(path=output_path)
    os.remove(fname)


def load_data_100k_np(path='./', delimiter='\t'):
    ''' Loading the dataset MovieLens into the predictor'''

    # Load raw data
    train = np.loadtxt(path+'u1.base', skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(path+'u1.test', skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:, 0]).size  # num of users
    n_m = np.unique(total[:, 1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    # Rating matrix
    train_rating = np.zeros((n_m, n_u), dtype='float32')
    test_rating = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        #            item_id         usr_id          rating
        train_rating[train[i, 1]-1, train[i, 0]-1] = train[i, 2]

    for i in range(n_test):
        #            item_id         usr_id          rating
        test_rating[test[i, 1]-1, test[i, 0]-1] = test[i, 2]

    # Masks, indicating non-zero entries
    train_mask = np.greater(train_rating, 1e-12).astype('float32')
    test_mask = np.greater(test_rating, 1e-12).astype('float32')

    return n_m, n_u, train_rating, train_mask, test_rating, test_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Zip archive dataset loader',
        description='Simple script for downloading zip dataset and unzipping it to specified folder'
    )

    parser.add_argument('-u', '--url', required=True, type=str, help='URL of zip file to download from')
    parser.add_argument('-p', '--path', type=str, default='./', help='Output path for the data')

    args = parser.parse_args()

    download_dataset(url=args.url, output_path=args.path)
