import tensorflow.keras as keras
import tensorflow.keras.backend as K


def nn_candidate_generation_multiclass(embedding_pairs, num_items):
    """Return a NN with both regular augmentation and concatenated embeddings"""
    input_layers, embedding_layers = [elem for pair in embedding_pairs for elem in pair.input_layers],\
                                     [elem for pair in embedding_pairs for elem in pair.embedding_layers]
    concat = keras.layers.Concatenate()(embedding_layers)
    layer_1 = keras.layers.Dense(64, activation='relu', name='layer1')(concat)
    output = keras.layers.Dense(num_items, activation='softmax', name='out')(layer_1)
    model = keras.models.Model(input_layers, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def nn_candidate_generation_binary(embedding_pairs):
    """Return a NN with both regular augmentation and concatenated embeddings"""
    input_layers, embedding_layers = [elem for pair in embedding_pairs for elem in pair.input_layers],\
                                     [elem for pair in embedding_pairs for elem in pair.embedding_layers]
    concat = keras.layers.Concatenate()(embedding_layers)
    layer_1 = keras.layers.Dense(64, activation='relu', name='layer1')(concat)
    output = keras.layers.Dense(1, activation='sigmoid', name='out')(layer_1)
    model = keras.models.Model(input_layers, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def nn_ranking(numeric_feature_count, embedding_pairs):
    """Return a NN with both regular augmentation and concatenated embeddings"""
    numeric_features_input = keras.layers.Input(shape=[numeric_feature_count])
    input_layers = [numeric_features_input] + [elem for pair in embedding_pairs for elem in pair.input_layers]
    pre_concat_layers = [numeric_features_input] + [elem for pair in embedding_pairs for elem in pair.embedding_layers]

    concat = keras.layers.Concatenate()(pre_concat_layers)
    layer_1 = keras.layers.Dense(64, activation='relu', name='layer1')(concat)
    output = keras.layers.Dense(1, kernel_initializer='lecun_uniform', name='out')(layer_1)
    model = keras.models.Model(input_layers, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
