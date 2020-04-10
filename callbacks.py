import tensorflow as tf
import numpy as np

from model import ConvertModel

class SaveProductionModel(tf.keras.callbacks.Callback):

    def __init__(self, patience=2):
        super(SaveProductionModel).__init__()
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.wait = 0

        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
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
        pass
