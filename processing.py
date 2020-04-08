import tensorflow as tf



def pre_transform(inputs):
    target_size = (32, 32, 3)
    samples = tf.divide(inputs, 255.0)
    samples = tf.image.resize(samples, size=(target_size[0], target_size[1]))
    samples = tf.reshape(samples, (-1, target_size[0], target_size[1], target_size[2]))
    return samples

def post_transform(inputs, **kwargs):
    path_class_labels = 'labels.txt'
    labels = tf.io.gfile.GFile(path_class_labels).read().split('\n')
    labels = tf.constant(labels)
    class_int = tf.argmax(inputs, axis=1)
    class_name = tf.gather(labels, class_int)
    class_name = tf.strings.regex_replace(class_name, "\t", "")
    print(class_name)
    return class_name

class Preprocess(tf.keras.layers.Layer):

    def __init__(self, target_size=(32, 32, 3)):
        super().__init__()
        self.resize_dims = (target_size[0], target_size[1])
        self.reshape_dims = (-1, target_size[0], target_size[1], target_size[2])

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        samples = tf.divide(inputs, 255.0)
        samples = tf.image.resize(samples, size=self.resize_dims)
        samples = tf.reshape(samples, self.reshape_dims)
        return samples

class Postprocess(tf.keras.layers.Layer):

    def __init__(self, path_class_labels='labels.txt'):
        super().__init__()
        self.labels = tf.constant(
            tf.io.gfile.GFile(
                path_class_labels
            ).read().split('\n')
        )

    def call(self, inputs, **kwargs):
        class_int = tf.argmax(inputs, axis=1)
        class_name = tf.gather(self.labels, class_int)
        class_name = tf.strings.regex_replace(class_name, "\t", "")
        return class_name

#
# def preprocessing_func(target_size=(32, 32, 3)):
#     width, height, n_dims = target_size
#
#     def transform(samples):
#         samples = tf.divide(samples, 255.0)
#         samples = tf.image.resize(samples, size=(width, height))
#         samples = tf.reshape(samples, (-1, width, height, n_dims))
#         return samples
#
#     return transform

def post_processing_func(path_class_labels='labels.txt'):
    labels = tf.io.gfile.GFile(path_class_labels).read().split('\n')

    def transform(predictions):
        class_int = tf.argmax(predictions, axis=1)
        label_index = tf.squeeze(class_int)
        class_name = labels[label_index]
        return class_name
    return transform
