#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt

# globals

# 28x28 images of hand-written digits
mnist = tf.keras.datasets.mnist

# get train and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# create neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=3)


# test the model
loss, accuracy = model.evaluate(x_test, y_test)
print(loss, accuracy)

