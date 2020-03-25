import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


from mnist_util import split_train_validation

mnist = tf.keras.datasets.mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()

train_image = (train_image > 10).astype(int)
test_image = (test_image > 10).astype(int)


train_image, train_label, validation_image, validation_label\
= split_train_validation(train_image, train_label, 80)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10)
])


# model.summary()



model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


model.fit(
    train_image, train_label,
    epochs = 5,
    validation_data=(validation_image, validation_label),
    verbose = 1
)



# predict = model(test_image[:25])
# predict_list = np.argmax(tf.nn.softmax(predict), -1)
#
#
#
#
# show_number = 25
# plt.figure(figsize=(25/2.54, 18/2.54))
# for i in range(show_number):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_image[i])
#     plt.xlabel(predict_list[i])
#     plt.ylabel(test_label[i], rotation=0)
#
# plt.show()



#
