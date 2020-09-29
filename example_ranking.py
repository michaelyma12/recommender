import sys
sys.path.append('/Users/michaelma/rush/recommender')

from modelling.transformations import *
from modelling.nn_utils import *
from modelling.recommender import *
from dataclasses import dataclass
from typing import List
from tensorflow.keras.callbacks import TensorBoard
from pickle import load
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import ml_metrics
import tensorflow as tf
import os


# select sample for model
model_dir = 'models/luxury-beauty'
ranking_dir = os.path.join(model_dir, 'ranking')
ratings = pd.read_csv('data/ratings/luxury-beauty.csv', names=['product_id', 'user_id', 'rating', 'timestamp'])
ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings[:100000]

# load encoders
user_id_encoder = load(open(os.path.join(model_dir, 'candidate-generation', 'user_id_encoder.pkl'), 'rb'))
product_id_encoder = load(open(os.path.join(model_dir, 'candidate-generation', 'product_id_encoder.pkl'), 'rb'))
ratings['user_id'] = ratings['user_id'].apply(lambda x: user_id_encoder[x])
ratings['product_id'] = ratings['product_id'].apply(lambda x: product_id_encoder[x])

# add some date features
ratings['weekday'] = ratings['date'].dt.weekday
ratings['month'] = ratings['date'].dt.month


def rolling_means(data, group_cols, mean_cols):
    """Compute expanding means given a set of group keys and operand columns"""
    for col in mean_cols:
        data['rolling_mean_' + col] = data.groupby(group_cols)[col].apply(lambda x: x.expanding().mean().shift())
    return data


def rolling_std(data, group_cols, mean_cols):
    """Compute expanding means given a set of group keys and operand columns"""
    for col in mean_cols:
        data['rolling_std_' + col] = data.groupby(group_cols)[col].apply(lambda x: x.expanding().std().shift())
    return data


def most_frequent(x): return x.value_counts().index[0]


most_frequent.__name__ = 'mode'
user_profiles = ratings.groupby(['user_id']).agg({
    'weekday': [most_frequent],
})
user_profiles.columns = ['_'.join(col) for col in user_profiles.columns.ravel()]

# compute rolling profiles
rolling_data = rolling_means(ratings, ['user_id'], ['rating'])
rolling_data = rolling_std(rolling_data, ['user_id'], ['rating'])
rolling_data['rolling_count'] = rolling_data.groupby(['user_id']).cumcount()
rolling_data['rolling_min'] = rolling_data.groupby(['user_id'])['rating'].cummin()
rolling_data['rolling_max'] = rolling_data.groupby(['user_id'])['rating'].cummax()
rolling_data.dropna(inplace=True)

# add review data
reviews = pd.read_json('data/reviews/luxury-beauty.json', lines=True)


# construct model data
numeric_features = [col for col in rolling_data.columns if 'rolling_' in col]
univalent_features = ['user_id']
shared_embeddings = SharedEmbeddingSpec('product_id', ['product_id'], ['touched_product_id'])
recommender_data = RecommenderData(rolling_data, user_col='user_id', item_col='product_id',
                                   target_col='rating', numeric_features=numeric_features,
                                   univalent_features=univalent_features,
                                   shared_features=[shared_embeddings])

recommender_data.get_touched_items('product_id')
recommender_data.segment_data()
recommender_data.build_embedding_layers()
ranking_inputs = recommender_data.build_model_inputs(recommender_data.x_train)

# build neural net
numeric_feature_count = len(numeric_features)
ranking_model = nn_ranking(numeric_feature_count, recommender_data.embedding_pairs)
ranking_model.fit(ranking_inputs, recommender_data.y_train)
print('Recommendations served!')

