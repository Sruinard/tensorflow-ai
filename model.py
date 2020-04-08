import tensorflow as tf
from custom_layers import Backbone

class CustomClassifier(tf.keras.models.Model):
    def __init__(self, n_layers, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.backbone = Backbone(n_layers=3)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(64, activation='relu') or _ in range(n_layers)]
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = self.backbone(x)
        x = self.flatten(x)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.dropout(x)
        x = self.dense(x)
        return self.out(x)

class ConvertModel(tf.keras.models.Model):

    def __init__(self, preprocessing_func, model, post_processing_func):
        super().__init__()
        self.preprocessing_func = preprocessing_func
        self.model = model
        self.post_processing_func = post_processing_func

    def call(self, inputs):
        x = self.preprocessing_func(inputs)
        x = self.model(x)
        x = self.post_processing_func(x)
        return x