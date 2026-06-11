import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise import dump

print("Loading data...")
df = pd.read_csv('netflix_recommender/data/processed/netflix_svd_subset.csv')
titles_df = pd.read_csv('netflix_recommender/data/raw/movie_titles.csv', encoding='ISO-8859-1', header=None, names=['movie_id', 'year', 'title'], on_bad_lines='skip')
title_map = dict(zip(titles_df['movie_id'], titles_df['title']))

print("Training Item-CF...")
user_ids = df['user_id'].astype('category')
movie_ids = df['movie_id'].astype('category')

user_idx = user_ids.cat.codes
movie_idx = movie_ids.cat.codes

user_map = dict(enumerate(user_ids.cat.categories))
movie_map = dict(enumerate(movie_ids.cat.categories))
rev_user_map = {v: k for k, v in user_map.items()}
rev_movie_map = {v: k for k, v in movie_map.items()}

sparse_user_item = csr_matrix((df['rating'], (user_idx, movie_idx)), shape=(len(user_map), len(movie_map)))
item_similarity = cosine_similarity(sparse_user_item.T)
np.fill_diagonal(item_similarity, 0)

print("Training SVD...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
trainset = data.build_full_trainset()
svd_model = SVD(n_factors=50, n_epochs=20, random_state=42)
svd_model.fit(trainset)

print("Saving models to netflix_recommender/models/...")
os.makedirs('netflix_recommender/models', exist_ok=True)

# Save SVD
dump.dump('netflix_recommender/models/svd_model.pkl', algo=svd_model)

# Save Item-CF and mappings
with open('netflix_recommender/models/item_cf_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'item_similarity': item_similarity,
        'user_map': user_map,
        'movie_map': movie_map,
        'rev_user_map': rev_user_map,
        'rev_movie_map': rev_movie_map,
        'title_map': title_map,
        'sparse_user_item': sparse_user_item
    }, f)

print("Models saved successfully!")
