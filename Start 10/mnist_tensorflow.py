import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

tf.config.experimental.set_visible_devices([], 'GPU')

from mnist_util import split_train_validation, load_real_data

def plot_image(predictions_array, true_label, img, predicted_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if predicted_label == true_label:
        color = 'red'
    else:
        color = 'black'

    plt.xlabel("{} {:2.0f}% ({})".format(
        predicted_label,
        100 * np.max(predictions_array),
        true_label),
        color=color
        )

def plot_bar(predictions_array, true_label, predicted_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='gray')
    plt.ylim([0, 1])

    thisplot[predicted_label].set_color('black')
    thisplot[true_label].set_color('red')


mnist = tf.keras.datasets.mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()

train_image = (train_image > 10).astype(int)
test_image = (test_image > 10).astype(int)


train_image, train_label, validation_image, validation_label\
= split_train_validation(train_image, train_label, 100)


train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))


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
    train_dataset.batch(64),
    epochs=50,
    verbose=1
)

model.evaluate(
    test_image, test_label,
    verbose=1
)


real_data = np.asarray(load_real_data('./real_img/', 10))
real_data_label = [2, 3, 7, 6, 3, 9, 0, 5, 8, 9]


predict = model(real_data)
predict = tf.math.softmax(predict)
predict_list = np.argmax(predict, -1)



num_rows = 4
num_cols = 3
num_images = 10
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(predict[i], real_data_label[i], real_data[i], predict_list[i])
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_bar(predict[i], real_data_label[i], predict_list[i])

plt.tight_layout()
plt.show()



#
