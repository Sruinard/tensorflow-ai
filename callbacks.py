""" File for writing custom keras callbacks """
# pylint: disable=W0201

import tensorflow as tf
import numpy as np

from model import ConvertModel

class SaveProductionModel(tf.keras.callbacks.Callback): # pylint: disable=R0902
    """save the best model for use in production"""

    def __init__(
            self,
            pre_processing,
            post_processing,
            data_batch,
            path_to_store_model,
            patience=2
    ):
        """

        Args:
            pre_processing: function to include in model
            post_processing: function to include in model
            data_batch: train batch to init shapes
            path_to_store_model: where to save production model
            patience: time to wait before stop training and save production model
        """
        super(SaveProductionModel).__init__()
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.data_batch = data_batch
        self.path_to_store_model = path_to_store_model
        self.patience = patience

    def on_train_begin(self, logs=None):
        """
        action to take when start training
        Args:
            logs: model logs for accessing attributes
        """

        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """
        check what best model weights are and determine to stop training
        Args:
            epoch: current epoch number
            logs: model logs for accessing attributes
        """

        current_loss = logs['loss']

        if current_loss < self.best:
            self.wait = 0
            self.best = current_loss
            self.best_model_weights = self.model.get_weights()

        else:
            self.wait += 1
            if self.wait > self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_model_weights)


    def on_train_end(self, logs=None):
        """ TODO load best model weights and save """
        # TODO get best weights
        model = ConvertModel(self.pre_processing, self.model, self.post_processing)
        model.predict(self.data_batch)
        model.save(self.path_to_store_model)
