import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train and x_test = array of images
# y_train and y_test = array of labels

print(len(x_train))
print(len(x_test))

print(y_train[0])
plt.imshow(x_train[0], cmap='gray')