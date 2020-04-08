import json
import requests
import numpy as np
import tensorflow as tf


def expand_to_batch_dims(data):
    if len(data.shape) == 3:
        data = np.expand_dims(data, 0)

    return data

def rest_request(data, url=None):
    data = data.tolist()
    data = expand_to_batch_dims(data)

    if url is None:
        url = 'http://localhost:8501/v1/models/mnist:predict'
    payload = json.dumps({"instances": data})
    response = requests.post(url, payload)
    return response



(_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
labels = tf.io.gfile.GFile('labels.txt').read().split('\n')
print(labels)
for index in range(10):
    data = x[index]
    print([labels[pred_index] for pred_index in y[index]])
    print(labels[y[index][0]])
    # print(unscaled_x.max())

    rs = rest_request(data)
    print(rs.json())