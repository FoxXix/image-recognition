import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train and x_test = array of images
# y_train and y_test = array of labels

print(len(x_train))
print(len(x_test))

print(y_train[0])
plt.imshow(x_train[0], cmap='gray')

class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__()

        # Conv2D(filters, kernel_size, activation)
        self.conv1 = Conv2D(32, 3, activation='relu')
        # Flatten() - 2d array to 1d array
        self.flatten = Flatten()
        # Dense(neurons, activation) - neural network layer
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


model = MNISTModel()
model(...)

