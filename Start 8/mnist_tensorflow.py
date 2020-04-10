import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mnist_util import split_train_validation, load_real_data

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
    epochs=5,
    validation_data=(validation_image, validation_label),
    verbose=1
)

model.evaluate(
    test_image, test_label,
    verbose=1
)


real_data = np.asarray(load_real_data('./real_img/', 10))

real_data_label = [2, 3, 7, 6, 3, 9, 0, 5, 8, 9]

predict = model(real_data)
predict_list = np.argmax(tf.nn.softmax(predict), -1)



show_number = 10
plt.figure(figsize=(25/2.54, 18/2.54))
for i in range(show_number):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(real_data[i])
    plt.xlabel(predict_list[i])
    plt.ylabel(real_data_label[i], rotation=0)

plt.show()



#
