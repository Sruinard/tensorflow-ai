import json
import requests
import numpy as np
import tensorflow as tf


def rest_request(data, url=None):
    data = data.tolist()
    if url is None:
        url = 'http://localhost:8501/v1/models/mnist:predict'
    payload = json.dumps({"instances": [data]})
    response = requests.post(url, payload)
    return response

(_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
labels = tf.io.gfile.GFile('labels.txt').read().split('\n')

for index in range(10):
    unscaled_x = np.reshape(x[index].copy(), (32, 32, 3))
    # x = preprocessing_func(x)
    data = x[index]

    print(labels[y[index][0]])
    # print(unscaled_x.max())

    rs = rest_request(unscaled_x)
    print(rs.json())