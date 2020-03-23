import tensorflow as tf

mnist = tf.keras.datasets.mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()

print(train_image[0])




#
