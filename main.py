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