import ml_metrics
import numpy as np
import pandas as pd


def average_precision(sorted_predictions, holdout_items, k=5):
    """compute average precision for a single user"""
    predicted_items = [pair[0] for pair in sorted_predictions]
    apk = ml_metrics.apk(holdout_items, predicted_items, k=k)
    frame = pd.DataFrame({'user_id': [user_recommendations['user_id'].values[0]], 'average_precision': [apk]})
    return frame

def mean_average_precision(val_frame, touched_dict, k=5):
    """take mean of all user average precisions"""
    average_precisions = val_frame.groupby(['user_id']).apply(lambda x: average_precision(x, touched_dict[x.name], k=k))
    return np.mean(average_precisions['average_precision'])
