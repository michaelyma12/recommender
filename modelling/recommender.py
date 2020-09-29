from modelling.transformations import *
from modelling.nn_utils import *

from dataclasses import dataclass
from typing import List
from ast import literal_eval
from sys import stdout

import os
import ml_metrics


@dataclass
class SharedEmbeddingSpec:
    """class to store shared embedding specifications"""
    name: str
    univalent: List[str]
    multivalent: List[str]


class RecommenderData(object):
    """input training data for recommender"""
    def __init__(self, model_frame=None, user_col='user_id', item_col='product_id', target_col='target',
                 numeric_features=[], univalent_features=[], multivalent_features=[],
                 shared_features=[]):
        # model components
        self.model_frame = model_frame
        self.user_col = user_col
        self.target_col = target_col
        self.item_col = item_col
        self.numeric_features = numeric_features
        self.univalent_features = univalent_features
        self.multivalent_features = multivalent_features
        self.shared_features = shared_features

        # training components
        self.x_train, self.y_train, self.x_val, self.y_val = None, None, None, None
        self.train_idx, self.val_idx = None, None

    def get_touched_items(self, item_col, window=10):
        """get rolling sets of touched items"""
        self.model_frame = self.model_frame.groupby(self.user_col).apply(lambda x: rolling_set(x, item_col, window))

    def segment_data(self):
        """prepare x and y matrices"""
        shared_features = [component for elem in self.shared_features for component in elem.univalent + elem.multivalent]
        features = self.model_frame[self.numeric_features + self.univalent_features + self.multivalent_features + shared_features]
        univalent_shared = [component for elem in self.shared_features for component in elem.univalent]
        target = self.model_frame[self.target_col]

        self.feature_indices = dict([(col, i) for i, col in enumerate(features.columns)])
        self.embedding_max_values = dict(features[self.univalent_features + self.multivalent_features + univalent_shared].nunique())
        self.embedding_dimensions = dict([(key, 20) for key, value in self.embedding_max_values.items()])

        x, y = features.values, target.values
        strat_shuffle = StratifiedShuffleSplit()
        train_idx, val_idx = strat_shuffle.split(x, y).__next__()
        self.x_train, self.y_train, self.x_val, self.y_val = x[train_idx], y[train_idx], x[val_idx], y[val_idx]
        self.train_idx, self.val_idx = train_idx, val_idx

    def build_embedding_layers(self):
        """build embedding layers for keras model"""
        self.embedding_pairs = [EmbeddingPair(embedding_name=feature,
                                              embedding_dimension=self.embedding_dimensions[feature],
                                              embedding_max_val=self.embedding_max_values[feature])
                                for feature in self.univalent_features] + \
                               [EmbeddingPair(embedding_name=feature,
                                              embedding_dimension=self.embedding_dimensions[feature],
                                              embedding_max_val=self.embedding_max_values[feature],
                                              valence='multivalent')
                                for feature in self.multivalent_features] + \
                               [EmbeddingPair(embedding_name=feature.name,
                                              embedding_dimension=self.embedding_dimensions[feature.name],
                                              embedding_max_val=self.embedding_max_values[feature.name],
                                              valence='shared', shared_embedding_spec=feature)
                                for feature in self.shared_features]

    def build_model_inputs(self, x):
        """return model inputs"""
        inputs = []
        numeric_indices = [self.feature_indices[feature] for feature in self.numeric_features]
        if numeric_indices: inputs.append(x[:, numeric_indices].astype(np.float32))
        
        for feature in self.univalent_features:
            inputs.append(x[:, self.feature_indices[feature]].astype(np.float32))

        for feature in self.multivalent_features:
            inputs.append(pad_sequences_batched(x, self.feature_indices[feature]).astype(np.float32))

        for feature in self.shared_features:
            for uni_feature in feature.univalent:
                inputs.append(x[:, self.feature_indices[uni_feature]].astype(np.float32))
            for multi_feature in feature.multivalent:
                inputs.append(pad_sequences_batched(x, self.feature_indices[multi_feature]).astype(np.float32))
        return inputs


class EmbeddingPair(object):
    """a pair of embedding inputs and layers"""
    def __init__(self, embedding_name, embedding_dimension, embedding_max_val, valence='univalent',
                 shared_embedding_spec=None):
        self.embedding_name = embedding_name
        self.embedding_dimension = embedding_dimension
        self.embedding_max_val = embedding_max_val
        self.input_layers = []
        self.embedding_layers = []
        self.valence = valence
        self.shared_embedding_spec = shared_embedding_spec

        if valence == 'univalent':
            self.build_univalent_layer()
        elif valence == 'multivalent':
            self.build_multivalent_layer()
        elif valence == 'shared':
            self.build_shared_layer(shared_embedding_spec)

    def build_univalent_layer(self):
        """build univalent embedding"""
        cat_id = keras.layers.Input(shape=(1,), name="input_" + self.embedding_name, dtype='int32')
        embeddings = keras.layers.Embedding(input_dim=self.embedding_max_val + 1,
                                            output_dim=int(self.embedding_dimension),
                                            name=self.embedding_name)(cat_id)
        embedding_vector = keras.layers.Flatten(name='flatten_' + self.embedding_name)(embeddings)
        self.input_layers.append(cat_id)
        self.embedding_layers.append(embedding_vector)

    def build_multivalent_layer(self):
        """build multivalent embedding"""
        cat_list = keras.layers.Input(shape=(None,), name='input_' + self.embedding_name, dtype='int32')
        embeddings = keras.layers.Embedding(input_dim=self.embedding_max_val + 2,
                                            output_dim=int(self.embedding_dimension),
                                            name=self.embedding_name + "_embedding", mask_zero=True)
        embeddings_avg = keras.layers.Lambda(lambda x: K.mean(x, axis=1), name=self.embedding_name + "_embeddings_avg")
        multivalent_vec = embeddings(cat_list)
        multivalent_avg = embeddings_avg(multivalent_vec)
        self.input_layers.append(cat_list)
        self.embedding_layers.append(multivalent_avg)

    def build_shared_layer(self, shared_embedding_spec):
        """build shared embedding inputs"""
        embeddings = keras.layers.Embedding(input_dim=self.embedding_max_val + 2,
                                            output_dim=int(self.embedding_dimension),
                                            name=self.embedding_name + "_embedding", mask_zero=True)
        embeddings_avg = keras.layers.Lambda(lambda x: K.mean(x, axis=1), name=self.embedding_name + "_embeddings_avg")

        for feature in shared_embedding_spec.univalent:
            shared_cat_id = keras.layers.Input(shape=(1,), name="input_" + feature, dtype='int32')
            shared_univalent_vec = embeddings(shared_cat_id)
            shared_univalent_avg = embeddings_avg(shared_univalent_vec)
            self.input_layers.append(shared_cat_id)
            self.embedding_layers.append(shared_univalent_avg)

        for feature in shared_embedding_spec.multivalent:
            shared_cat_list = keras.layers.Input(shape=(None,), name='input_' + feature, dtype='int32')
            shared_multivalent_vec = embeddings(shared_cat_list)
            shared_multivalent_avg = embeddings_avg(shared_multivalent_vec)
            self.input_layers.append(shared_cat_list)
            self.embedding_layers.append(shared_multivalent_avg)


class CandidateGenerationDataPandas(RecommenderData):

    def __init__(self, model_frame, user_col='user_id', item_col='product_id', target_col='target',
                 numeric_features=[], univalent_features=[], multivalent_features=[],
                 shared_features=[], initialize=True):
        """create a datum specifically for the candidate generation networkS"""
        super().__init__(model_frame=model_frame, user_col=user_col, item_col=item_col,
                         target_col=target_col, numeric_features=numeric_features,
                         univalent_features=univalent_features, multivalent_features=multivalent_features,
                         shared_features=shared_features)

        # perform basic processing
        if initialize:
            print('Getting touched items ...')
            self.get_touched_items(self.item_col)
            print('Beginning negative sampling ...')
            self.negative_sample(item_col=self.item_col, touched_col='touched_{}'.format(self.item_col))

    def negative_sample(self, item_col, touched_col):
        """negative sample to prevent folding in embedding layers"""
        temp = self.model_frame.copy()
        temp['negatives'] = temp.apply(
            lambda x: negative_sampling(np.concatenate([np.array(x[touched_col]), np.array([x[item_col]])], axis=0),
                                        temp[item_col].nunique(), sample_size=3), axis=1)

        temp = temp.explode(column='negatives')
        del temp[item_col]
        temp.rename(columns={'negatives': item_col}, inplace=True)
        temp[self.target_col] = 0

        temp = pd.concat([self.model_frame, temp], axis=0)
        temp.reset_index(inplace=True)
        self.model_frame = temp


class CandidateGenerationData(RecommenderData):

    def __init__(self, user_col='user_id', item_col='product_id', target_col='target', numeric_features=[],
                 univalent_features=[], multivalent_features=[], shared_features=[]):
        super().__init__(user_col=user_col, item_col=item_col, target_col=target_col, numeric_features=numeric_features,
                         univalent_features=univalent_features, multivalent_features=multivalent_features,
                         shared_features=shared_features)

    def load_train_data(self, model_storage):
        """load data"""
        self.feature_indices = model_storage.load_pickle('feature_indices.pkl')
        self.feature_types = model_storage.load_pickle('feature_types.pkl')
        self.holdout_types = model_storage.load_pickle('holdout_types.pkl')
        self.embedding_max_values = model_storage.load_pickle('embedding_max_values.pkl')
        self.embedding_dimensions = dict([(key, 20) for key, value in self.embedding_max_values.items()])

        stdout.write('DEBUG: Loading holdout frame and categorical encoders ... \n')
        self.user_encoder = model_storage.load_pickle('user_id_encoder.pkl')
        self.product_encoder = model_storage.load_pickle('product_id_encoder.pkl')
        self.holdout_frame = self.fit_feature_types(
            pd.read_csv(os.path.join(model_storage.bucket_uri, model_storage.model_path, 'holdout.csv')),
            self.holdout_types
        ) if model_storage.gcs else self.fit_feature_types(
            pd.read_csv(os.path.join(model_storage.local_path, 'holdout.csv')), self.holdout_types
        )

        stdout.write('DEBUG: Loading train and validation dataframes ... \n')
        train_df = pd.read_csv(os.path.join(model_storage.bucket_uri, model_storage.model_path, 'train.csv')) if\
            model_storage.gcs else \
            pd.read_csv(os.path.join(model_storage.local_path, 'train.csv'))
        train_matrix = self.fit_feature_types(train_df, self.feature_types).values

        val_df = pd.read_csv(os.path.join(model_storage.bucket_uri, model_storage.model_path, 'validation.csv')) if\
            model_storage.gcs \
            else pd.read_csv(os.path.join(model_storage.local_path, 'validation.csv'))
        val_matrix = self.fit_feature_types(val_df, self.feature_types).values

        y_index = self.feature_indices[self.target_col]
        x_indices = [i for col, i in self.feature_indices.items() if col != self.target_col]
        self.x_train, self.y_train = train_matrix[:, x_indices], train_matrix[:, y_index].astype(np.float32)
        self.x_val, self.y_val = val_matrix[:, x_indices], val_matrix[:, y_index].astype(np.float32)


    @staticmethod
    def fit_feature_types(pd_frame, feature_types):
        """convert feature types from string to original sparksql schema"""
        for feature in pd_frame.columns:
            feature_type =feature_types[feature]
            if 'ArrayType' in feature_type:
                if type(pd_frame[feature][0]) != list: pd_frame[feature] = pd_frame[feature].apply(literal_eval)
            elif feature_type == 'IntegerType':
                pd_frame[feature] = pd_frame[feature].astype(int)
            elif feature_type == 'StringType':
                pd_frame[feature] = pd_frame[feature].astype(str)
        return pd_frame


class CandidateGenerator(object):

    def __init__(self, cg_data, cg_model):
        """store neural net and make recommendations"""
        self.cg_data = cg_data
        self.cg_model = cg_model

    def construct_recommendation_frame(self, user_series):
        """construct cartesian frame to make predictions for every product given a single user"""
        index = pd.MultiIndex.from_product([[user_series['user_id']], list(self.cg_data.product_encoder.values())],
                                           names=['user_id', 'product_id'])
        recommendation_frame = pd.DataFrame(index=index)
        recommendation_frame['touched_product_id'] = [user_series['touched_product_id']] * len(recommendation_frame)
        recommendation_frame['liked_product_id'] = [user_series['liked_product_id']] * len(recommendation_frame)
        recommendation_frame['disliked_product_id'] = [user_series['disliked_product_id']] * len(recommendation_frame)
        recommendation_frame.reset_index(inplace=True)
        return recommendation_frame

    def recommend_for_user(self, user_series):
        """make recommendations for a user"""
        recommendation_frame = self.construct_recommendation_frame(user_series)
        recommendation_inputs = self.cg_data.build_model_inputs(recommendation_frame.values)
        recommendations = list(
            zip(recommendation_frame['product_id'].values, list(map(lambda x: x[0], self.cg_model.predict(recommendation_inputs))))
        )
        recommendations = sorted(recommendations, key=lambda x: -x[1])
        return recommendations

    def get_average_precisions(self, k=100):
        """construct average precision frame"""
        precisions = []
        for i, user_series in self.cg_data.holdout_frame.iterrows():
            stdout.write('Computing average precision for: ' + str(user_series['user_id']) + '\n')
            recommendations = self.recommend_for_user(user_series)
            recommended_items = list(map(lambda x: x[0], recommendations))
            avg_precision = ml_metrics.apk(user_series['holdout_product_id'], recommended_items, k=k)
            precisions.append({'user_id': user_series['user_id'], 'average_precision': avg_precision})
        return pd.DataFrame(precisions)

    def map_at_k(self, k=100):
        """evaluate mean average precision @ k, also return individual precisions for perusal"""
        average_precisions = self.get_average_precisions(k=k)
        mean_avg_precision = np.mean(average_precisions['average_precision'])
        return mean_avg_precision, average_precisions


