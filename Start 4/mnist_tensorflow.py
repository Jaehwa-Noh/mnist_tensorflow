import tensorflow as tf
import matplotlib.pyplot as plt

from mnist_util import split_train_validation

mnist = tf.keras.datasets.mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()

train_image = (train_image > 10).astype(int)
test_image = (test_image > 10).astype(int)


train_image, train_label, validation_image, validation_label\
= split_train_validation(train_image, train_label, 80)

print(len(train_image), len(train_label))
print(len(validation_image), len(validation_label))


# plt.figure(figsize=(25/2.54, 18/2.54))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_image[i])
#     plt.xlabel(train_label[i])
#
# plt.show()



#
