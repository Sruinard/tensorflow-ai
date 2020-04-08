import tensorflow as tf

class PreProcessTFLayer(tf.keras.layers.Layer):
    def __init__(self, preprocessing_func):
        super().__init__()
        self.preprocess = preprocessing_func

    def call(self, input):
        return self.preprocess(input)


class PostProcessTFLayer(tf.keras.layers.Layer):
    def __init__(self, post_processing_func):
        super().__init__()
        self.post_processing = post_processing_func

    def call(self, input):
        return self.post_processing(input)


class ExtendedConv(tf.keras.layers.Layer):

    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(n_filters, kernel_size)
        self.act = tf.keras.layers.ReLU()
        self.maxpool2d = tf.keras.layers.MaxPool2D()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.act(x)
        x = self.maxpool2d(x)
        return x


class Backbone(tf.keras.layers.Layer):

    def __init__(self, n_layers):
        super().__init__()
        self.backbone_layers = [
            ExtendedConv(64, 3) for _ in range(n_layers)
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.backbone_layers:
            x = layer(x)
        return x