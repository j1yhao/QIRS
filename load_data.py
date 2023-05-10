import math

import pandas as pd
import numpy as np
import random


def read_user():
    df_user = pd.read_csv('./data/ml-100k/u.user', sep='|',
                          names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    age_bins = [0, 18, 25, 35, 45, 50, 56, 100]
    age_labels = ['0-17', '18-24', '25-34', '25-44', '45-49', '50-55', '56-100']
    df_user['age_bin'] = pd.cut(df_user['age'], bins=age_bins, labels=age_labels)
    age_onehot = pd.get_dummies(df_user['age_bin'])
    gender_onehot = pd.get_dummies(df_user['gender'])
    occupation_onehot = pd.get_dummies(df_user['occupation'])
    user_onehot = pd.concat([df_user[['user_id']], age_onehot, gender_onehot, occupation_onehot], axis=1)

    filename = "./data/ml-100k/u.item"
    rnames = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv(filename, sep='|', header=None, names=rnames, engine='python', encoding='latin1')
    df_item.drop(['title', 'release_date', 'video_release_date', 'IMDb_URL'], axis=1, inplace=True)

    filename = "./data/ml-100k/u.data"
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    df_ratings = pd.read_table(filename, sep='\t', header=None, names=rnames, engine='python')
    df_ratings.drop(['timestamp'], axis=1, inplace=True)
    data = pd.merge(df_ratings, user_onehot, on='user_id')
    data = pd.merge(data, df_item, on='movie_id')
    print(data)
    np.save("FM_model.npy", data.values)


if __name__ == '__main__':
    read_user()

