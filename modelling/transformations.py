import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from pickle import dump
from collections import defaultdict, OrderedDict

from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.preprocessing.sequence import pad_sequences


def label_encode_categoricals(dataframe, columns, save_path):
    """Label encode a batch of categorical columns"""
    for col in columns:
        le = LabelEncoder()
        le.fit(dataframe[col])
        dataframe[col] = le.transform(dataframe[col])
        dump(le, open(save_path + "/label_encoded_" + col + ".pkl", 'wb'))
    return dataframe


def negative_sampling(pos_ids, num_items, sample_size=10):
    """negative sample for candidate generation. assumes pos_ids is ordered."""
    raw_sample = np.random.randint(0, num_items - len(pos_ids), size=sample_size)
    pos_ids_adjusted = pos_ids - np.arange(0, len(pos_ids))
    ss = np.searchsorted(pos_ids_adjusted, raw_sample, side='right')
    neg_ids = raw_sample + ss
    return neg_ids


def rolling_set(data, item_col='product_id', window_size=2):
    """function for groupby apply calls"""
    data['touched_{}'.format(item_col)] = [list(data.iloc[max(0, i-window_size):i][item_col].values) for i in range(len(data))]
    return data


def pad_sequences_batched(x, col_index, num_batches=10, pad_len=10):
    """pad column in numpy array in batches"""
    intervals = np.arange(0, x.shape[0] + x.shape[0] // num_batches, x.shape[0] // num_batches)
    final = []
    for i in range(len(intervals) - 1):
        print('Padding batch: ' + str(i))
        final.append(pad_sequences(x[intervals[i]:intervals[i + 1], col_index], padding='post', maxlen=pad_len))
    final = np.concatenate(final, axis=0)
    return final
