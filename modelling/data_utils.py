from google.cloud import storage
from pickle import loads, load

import os
import glob


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    """upload a local directory to gcs recursively"""
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
            blob = bucket.blob(os.path.join(gcs_path, local_file[1 + len(local_path):]))
            blob.upload_from_filename(local_file)


class ModelStorage(object):

    def __init__(self, bucket_name, model_path, gcs=True):
        """manage files across cloud & local storage"""
        self.bucket_name = bucket_name
        self.bucket_uri = 'gs://{}'.format(bucket_name)
        self.model_path = model_path
        self.local_path = model_path
        self.gcs = gcs

        if gcs:
            self.storage_client = storage.Client()
            self.storage_bucket = self.storage_client.get_bucket(bucket_name)

    def load_pickle(self, path):
        """read pickle file from google cloud storage or local storage"""
        if self.gcs:
            return loads(self.storage_bucket.blob(os.path.join(self.model_path, path)).download_as_string())
        else:
            return load(open(os.path.join(self.local_path, path), 'rb'))

    def save_file_gcs(self, file_path):
        """save file to gcs"""
        save_blob = self.storage_bucket.blob(os.path.join(self.model_path, file_path))
        save_blob.upload_from_filename(os.path.join(self.local_path, file_path))

    def save_model_gcs(self, model):
        """Save a model as yaml"""
        model_yaml = model.to_yaml()
        with open(os.path.join(self.local_path, 'model.yaml'), 'w') as yaml_file:
            yaml_file.write(model_yaml)
        model.save_weights(os.path.join(self.local_path, 'model.h5'))
        self.save_file_gcs('model.yaml')
        self.save_file_gcs('model.h5')

    def save_directory_gcs(self, directory_path):
        """Save a directory to gcs"""
        upload_local_directory_to_gcs(os.path.join(self.local_path, directory_path), self.storage_bucket,
                                      os.path.join(self.model_path, directory_path))



