import tensorflow as tf

mnist = tf.keras.datasets.mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()

print(tf.reduce_max(train_image))
print(tf.reduce_min(train_image))





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
