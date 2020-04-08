import tensorflow as tf
import numpy as np

from processing import pre_transform, post_transform, Preprocess, Postprocess#preprocessing_func,
from model import ConvertModel, CustomClassifier



# class PreProcessTFLayer(tf.keras.layers.Layer):
#     def __init__(self, preprocessing_func):
#         super().__init__()
#         self.preprocess = preprocessing_func
#
#     def call(self, input):
#         return self.preprocess(input)
#
# class PostProcessTFLayer(tf.keras.layers.Layer):
#     def __init__(self, post_processing_func):
#         super().__init__()
#         self.post_processing = post_processing_func
#
#     def call(self, input):
#         return self.post_processing(input)
#
# class ConvertModel(tf.keras.models.Model):
#
#     def __init__(self, preprocessing_func, model, post_processing_func):
#         super().__init__()
#         self.model = model
#         self.preprocessing_func = preprocessing_func
#         self.post_processing_func = post_processing_func
#
#     def call(self, inputs):
#         x = self.preprocessing_func(inputs)
#         x = self.model(x)
#         x = self.post_processing_func(x)
#         return x

# def preprocessing_func(samples, target_size):
#     width, height, n_dims = target_size
#     samples = tf.divide(samples, 255.0)
#     samples = tf.image.resize(samples, size=(width, height))
#     samples = tf.reshape(samples, (-1, width, height, n_dims))
#     return samples
#
# def post_processing_func(predictions):
#     class_int = tf.argmax(predictions, axis=1)
#     return class_int

def main(model_version):
    path = f'./my_model/{model_version}/'
    path_full = f'./my_full_model/{model_version}/'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    PreProcessingLayer = Preprocess()
    PostProcessingLayer = Postprocess()

    x_train = PreProcessingLayer(x_train)
    x_test = PreProcessingLayer(x_test)

    model = CustomClassifier(4, 10)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

    model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)
    model.save(path)

    inference_model = ConvertModel(PreProcessingLayer, model, PostProcessingLayer)
    inference_model.predict(x_test[:10])
    inference_model.save(path_full)


def fine_tune_full(model_version):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = preprocessing_func(x_train)
    x_test = preprocessing_func(x_test)
    path = f'./my_full_model/{model_version}/'
    model = tf.keras.models.load_model(path)
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test)

    path = f'./my_model/{model_version+1}/'
    model.save(path)
    print(y_test[:10])
    np.save('test_data', x_test[:10])

def fine_tune(model_version):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = preprocessing_func(x_train)
    x_test = preprocessing_func(x_test)
    path = f'./my_model/{model_version}/'
    model = tf.keras.models.load_model(path)
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    model.evaluate(x_test, y_test)

    path = f'./my_model/{model_version+1}/'
    model.save(path)
    print(y_test[:10])
    np.save('test_data', x_test[:10])


if __name__ == '__main__':
    #fine_tune_full(1)
    main(1)