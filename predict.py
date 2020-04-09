import json
import requests
import numpy as np
import tensorflow as tf


def expand_to_batch_dims(data):
    if len(data.shape) == 3:
        data = np.expand_dims(data, 0)

    return data

def rest_request(data, url=None):
    data = expand_to_batch_dims(data)
    data = data.tolist()

    if url is None:
        url = 'http://localhost:8501/v1/models/mnist:predict'
    payload = json.dumps({"instances": data})
    response = requests.post(url, payload)
    return response

def test_tflite(input_data):
    path = "./my_tflite_models/converted_model.tflite"

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_data = tf.cast(input_data, tf.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def main(is_tf_serving=False):
    (_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
    labels = tf.io.gfile.GFile('labels.txt').read().split('\n')

    for index in range(10):
        data = x[index:index+1]
        class_name = labels[y[index][0]]


        if is_tf_serving:
            print('======================')
            print("ground truth:", class_name)
            rs = rest_request(data)
            print(rs.json())
            print('======================')
        else:
            print('======================')
            print("ground truth:", class_name)
            prediction = test_tflite(data)
            print("predicted:", prediction[0].decode('utf-8'))
            print('======================')

if __name__ == '__main__':
    main()