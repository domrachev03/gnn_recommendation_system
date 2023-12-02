from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import pandas as pd

sys.path.append('benchmark/benchmark_models')
sys.path.append('src/data')

from load_dataset import load_data_100k_np
from GLocal_K import *

torch.manual_seed(1284)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_similar_user(user_data: dict, data_path: str = 'data/raw/ml-100k/') -> int:
    user = pd.read_csv(f'{data_path}/u.user', sep="|", encoding='latin-1', header=None)
    user.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']

    available_occupations = set(pd.unique(user['occupation']))
    if not user_data['occupation'] in available_occupations:
        user_data['occupation'] = 'other'

    similar_users = user[(user['gender'] == user_data['gender']) & (user['occupation'] == user_data['occupation'])]
    age_diff = np.abs(similar_users['age'] - user_data['age'])

    return similar_users.iloc[np.argmin(age_diff), 0]


def print_films_list(films_id: list, data_path: str = 'data/raw/ml-100k/'):
    item = pd.read_csv(f'{data_path}/u.item', sep="|", encoding='latin-1', header=None)
    genres = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    item.columns = ['id', 'title', 'release_date', 'video_release_date', 'url', *genres]

    print('='*80)
    print('Recommendations list')
    print('='*80)
    for i, id in enumerate(films_id):
        film = item[item['id'] == id]
        print(f'Film #{i+1}:')
        print(f'     {film["title"].item()}, {film["release_date"].item()}')
        print('     Genre: ', end="")
        for genre in genres:
            if film[genre].item():
                print(f"{genre}, ", end="")
        print()
        print(f'     Link: {film["url"].item()}')
        print('-'*80)


def get_user_data() -> dict:
    os.system('cls' if os.name == 'nt' else 'clear')
    print('Welcome to the film recommendation service!')
    print('Please, enter the information about yourself:')

    print('     Name & Surname: ', end="")
    input()

    print('     Gender (M/F): ', end="")
    gender = input()
    while gender.upper() != 'M' and gender.upper() != 'F':
        print('     Wrong input format. Please, enter either M(ale) or F(emale): ', end="")
        gender = input()
    gender = gender.upper()

    print('     Age: ', end="")
    age = input()
    while not age.isdigit():
        print('     Wrong input format. Please, enter age as positive number: ', end="")
        age = input()
    age = int(age)

    print("     Occupation: ", end="")
    occupation = input()

    print("Thank you for your answers. Now, please wait...")
    return {
        'age': age,
        'gender': gender,
        'occupation': occupation.lower()
    }


def glocal_k_inference(data_path='data/raw/ml-100k/', weights='models/glocal_k/best_model_rmse.pt', predict_new=False, k=5):
    user_data = get_user_data()
    similar_user_idx = get_similar_user(user_data)

    n_m, n_u, _, _, data, mask = load_data_100k_np(path=data_path, delimiter='\t')

    glocal_k = torch.load(weights, map_location=torch.device(device))

    x = torch.Tensor(data).double().to(device)

    # Running the model to predict
    glocal_k.eval()
    preds = glocal_k(x)[0].float().cpu().detach().numpy()

    if predict_new:
        mask = np.where(mask == 0, 1, 0)
        preds *= mask

    predicted_films_id = np.argpartition(preds[:, similar_user_idx], -k)[-k:]
    print_films_list(predicted_films_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Inference, as naive recomenndation system for new users',
        description='''The script collects data from users, finds the most familiar user in the database and suggests the films that he might like'''
    )

    parser.add_argument('-n', '--n_films', type=int, default=5, help="Amount of films to suggest")
    parser.add_argument('-b', '--benchmark', action='store_true', help="If specified, the benchmark model (GLocal-K) would be used")
    parser.add_argument('-p', '--path', type=str, default='data/raw/ml-100k/', help='Path to the data')
    parser.add_argument('-w', '--weights', type=str, default='models/glocal_k/best_model_rmse.pt', help='Directory to load weights.')
    parser.add_argument('--new', action='store_true', help="If specified, then the model would predict only the films that the original person hasn't seen. Meanwhile this makes little sense in the current scenario, this makes inference more reasonable.")
    args = parser.parse_args()
    if args.benchmark:
        glocal_k_inference(data_path=args.path, weights=args.weights, predict_new=args.new, k=args.n_films)
