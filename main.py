import numpy as np
import tensorflow as tf

from model import ConvertModel, CustomClassifier
from processing import Preprocess, Postprocess


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
    path = f'./my_model/{model_version}/'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    PreProcessingLayer = Preprocess()
    PostProcessingLayer = Postprocess()

    x_train = PreProcessingLayer(x_train)
    x_test = PreProcessingLayer(x_test)

    model = tf.keras.models.load_model(path)

    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)

    path = f'./my_model/{model_version + 1}/'
    model.save(path)

    path_full = f'./my_full_model/{model_version + 1}/'
    inference_model = ConvertModel(PreProcessingLayer, model, PostProcessingLayer)
    inference_model.predict(x_test[:10])
    inference_model.save(path_full)

def convert_to_tflite(model_version):
    path = f'./my_model/{model_version}/'

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    PreProcessingLayer = Preprocess()
    PostProcessingLayer = Postprocess()

    x_test = PreProcessingLayer(x_test)

    model = tf.keras.models.load_model(path)

    inference_model = ConvertModel(PreProcessingLayer, model, PostProcessingLayer)
    inference_model.predict(x_test[:10])

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    tflite_model = converter.convert()
    open("./my_tflite_models/converted_model.tflite", "wb").write(tflite_model)






if __name__ == '__main__':
    test_tflite()


    # main(1)
    # fine_tune_full(1)
    # convert_to_tflite(2)