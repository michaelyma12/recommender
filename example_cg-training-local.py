import sys
from sys import stdout
sys.path.append('/Users/michaelma/rush/recommender')

from dataclasses import dataclass
from typing import List
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import storage

import pandas as pd
import ml_metrics
import time
import tensorflow as tf
import os

from modelling.transformations import *
from modelling.nn_utils import *
from modelling.recommender import *
from modelling.data_utils import *
from modelling.evaluation import *

# load cloud storage, configure training and validation matrices
stdout.write('DEBUG: Reading data from local storage ...\n')
cg_storage = ModelStorage(bucket_name='recommender-amazon-1', model_path='models/luxury-beauty/candidate-generation')
shared_embeddings = SharedEmbeddingSpec(name='product_id',
                                        univalent=['product_id'],
                                        multivalent=['touched_product_id', 'liked_product_id', 'disliked_product_id'])
cg_data = CandidateGenerationData(univalent_features=['user_id'], shared_features=[shared_embeddings])
cg_data.load_train_data(cg_storage, gcs=False)

# begin model construction
stdout.write('DEBUG: Building model inputs ... \n')
class_weights = {0: 1, 1: 3}
cg_data.build_embedding_layers()
cg_inputs_train = cg_data.build_model_inputs(cg_data.x_train)
cg_inputs_val = cg_data.build_model_inputs(cg_data.x_val)
stdout.write('DEBUG: Listing available CPUs/GPUs ... \n')
stdout.write(str(device_lib.list_local_devices()))

stdout.write('DEBUG: Fitting model ... \n')
from tensorflow.keras.models import load_model
tensorboard_callback = TensorBoard(log_dir=os.path.join(cg_storage.local_path, 'logs'), histogram_freq=1,
                                   write_images=True)
keras_callbacks = [tensorboard_callback]
cg_model = nn_candidate_generation_binary(cg_data.embedding_pairs)
start = time.time()
cg_model.fit(cg_inputs_train, cg_data.y_train, class_weight=class_weights, epochs=6,
             callbacks=keras_callbacks, batch_size=512, validation_data=(cg_inputs_val, cg_data.y_val))
duration = time.time() - start
stdout.write('BENCHMARKING: Total training time was ' + str(duration) + '\n')

# save model locally
model_yaml = cg_model.to_yaml()
with open(os.path.join(cg_storage.local_path, 'model.yaml'), 'w') as yaml_file:
    yaml_file.write(model_yaml)
cg_model.save_weights(os.path.join(cg_storage.local_path, 'model.h5'))

# evaluate classification metrics for validation set
predictions = cg_model.predict(cg_inputs_val)
predictions_class = np.where(predictions > 0.5, 1, 0)
print('EVALUATION: Building classification report ... \n')
stdout.write(classification_report(cg_data.y_val, predictions_class))

# eval recommender
candidate_generator = CandidateGenerator(cg_data=cg_data, cg_model=cg_model)
k = int(len(cg_data.product_encoder.values()) / 50)
mean_avg_p, avg_p_frame = candidate_generator.map_at_k(k = k)
