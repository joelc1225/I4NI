#!/usr/bin/env python
# Try to follow the tutorial more exactly

import sys
from os import path

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

### Paths ###
PATH = dict()
PATH['repo'] = path.dirname(path.dirname(path.abspath((__file__))))
PATH['data'] = path.join(PATH['repo'], 'data')
PATH['ML-datasets'] = path.join(PATH['data'], 'ML-datasets')
PATH['sonar'] = path.join(PATH['data'], 'sonar-clean.csv')
PATH['nn'] = path.join(PATH['repo'], 'neural-networks')
PATH['graph'] = path.join(PATH['nn'], 'graph')
PATH['saved-session'] = path.join(PATH['nn'], 'saved-session.hmm?')

### Global Configuration ###
# number of INPUT dimensions
NUM_DIM = 60
# number of OUTPUT dimensions
NUM_DIM_OUTPUT = 1
NUM_CATEGORIES = 2
# even though there are 2 categories, there is just ONE output value which equals '0' for a rock and '1' for a mine
NUM_COLS = NUM_DIM + NUM_DIM_OUTPUT
# number neurons for each layer
NUM_NEURONS = [NUM_DIM, 60, 60, 60, 60, NUM_CATEGORIES]
# number of layers, including the input layer
NUM_LAYERS = 6
# the learning rate of the gradient descent optimizer.  i'm assuming this is the STEP that the variables can budge each epoch
# LEARNING_RATE = 0.3
LEARNING_RATE = 0.003
# tf.enable_eager_execution()
# models can easily blow up (i believe this causes the 'nan' and 'inf' values).  That's why i get numbers for TRAINING_EPOCHS <=3 and nan otherwise
TRAINING_EPOCHS = 10


def enforce_python_version():
	if sys.version_info[0] != 2 or sys.version_info[1] != 7:
		raise SystemExit('Please use Python version 2.7.')

def identity(x):
	# the (unary) identity function
	return x

def one_hot_encode(labels):
	num_labels = len(labels)
	num_unique_labels = len(np.unique(labels))
	encode = np.zeros((num_labels, num_unique_labels))
	encode[np.arange(num_labels), labels] = 1
	return encode

def read_dataset():
	# tf (on this machine at least) is using float32 as the default precision, so we need to use the same type
	df = pd.read_csv(PATH['sonar'], dtype=np.float32)
	assert len(df.columns) == NUM_COLS
	# features (input)
	feature = df[df.columns[0 : NUM_DIM]].values
	assert feature.shape[1] == NUM_DIM
	# labels (desired output)
	# y = df[df.columns[NUM_DIM : NUM_DIM + NUM_DIM_OUTPUT]]
	# shorthand of above, but returns differently (TODO: make above work)
	y = df[df.columns[60]]
	# assert y.shape[1] == NUM_DIM_OUTPUT
	# y.shape == (208,)
	encoder = LabelEncoder()
	encoder.fit(y)
	y = encoder.transform(y)
	desired_result = one_hot_encode(y)
	return (feature, desired_result)

WEIGHTS = {
	# each neuron in layer 1 has weights that it applies to neurons in layer 0
	1: tf.Variable(tf.truncated_normal([NUM_NEURONS[0], NUM_NEURONS[1]])),
	# each neuron in layer 2 has weights that it applies to neurons in layer 1
	2: tf.Variable(tf.truncated_normal([NUM_NEURONS[1], NUM_NEURONS[2]])),
	3: tf.Variable(tf.truncated_normal([NUM_NEURONS[2], NUM_NEURONS[3]])),
	4: tf.Variable(tf.truncated_normal([NUM_NEURONS[3], NUM_NEURONS[4]])),
	5: tf.Variable(tf.truncated_normal([NUM_NEURONS[4], NUM_NEURONS[5]])),
}
BIASES = {
	# each neuron in the first layer (NOT input layer) has a bias
	1: tf.Variable(tf.truncated_normal([NUM_NEURONS[1]])),
	# each neuron in layer 2 has a bias
	2: tf.Variable(tf.truncated_normal([NUM_NEURONS[2]])),
	3: tf.Variable(tf.truncated_normal([NUM_NEURONS[3]])),
	4: tf.Variable(tf.truncated_normal([NUM_NEURONS[4]])),
	5: tf.Variable(tf.truncated_normal([NUM_NEURONS[5]])),
}

def layer(x, weights, biases, activation_func):
	# print(x)
	# print(weights)
	dot_prods = tf.matmul(x, weights)
	sums = tf.add(dot_prods, biases)
	y = activation_func(sums)
	return y

def multilayer_perceptron(x, weights, biases):
	# for me a "layer" is a strip of nodes that hold values.  The product, summation, bias, and activation occurs BETWEEN layers
	# that output of layer 0 is `layer_values[0]`, the output of layer 1 is `layer_output[1]`, and so forth...
	layer_values = [None] * NUM_LAYERS
	# input layer (here x = feature)
	layer_values[0] = x
	# three hidden layers with sigmoid activation
	for i in range(1, 4):
		layer_values[i] = layer(layer_values[i - 1], weights[i], biases[i], tf.nn.sigmoid)
	# one hidden layer with RELU activation
	layer_values[4] = layer(layer_values[3], weights[4], biases[4], tf.nn.relu)
	# output later
	layer_values[5] = layer(layer_values[4], weights[5], biases[5], identity)
	assert all(vals is not None for vals in layer_values)
	return layer_values[-1]

def build_and_run_graph():
	# read data
	feature, desired_result = read_dataset()
	feature, desired_result = shuffle(feature, desired_result, random_state=1)
	# split into train and test data
	feature_train, feature_test, desired_result_train, desired_result_test = train_test_split(
			feature,
			desired_result,
			test_size=0.2,
			random_state=415)
	# print(type(feature))       # numpy ndarray

	# print(feature.shape)       # 208, 60
	# print(feature_train.shape) # 166, 60
	# print(feature_test.shape)  # 42, 60

	# print(desired_result.shape)              # 208, 2
	# print(desired_result_train.shape)        # 166, 2
	# print(desired_result_test.shape)         # 42, 2

	error_history = np.empty(shape=[1], dtype=float)

	result = multilayer_perceptron(feature, WEIGHTS, BIASES)
	difference = result - desired_result
	error = tf.reduce_sum(tf.square(difference))
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
	thang = optimizer.minimize(error)

	# RUN
	# run a graph
	init = tf.global_variables_initializer()
	# TODO: figure out how to save a session
	# saving session is optional, could comment it out
	# saver = tf.train.Saver()
	session = tf.Session()
	session.run(init)
	# was 'model_path' in demo
	# saver.restore(session, PATH['saved-session'])
	for epoch in range(TRAINING_EPOCHS):
		session.run([error, thang])
		# print('epoch: {} - error: {} - train accuracy: {}'.format(epoch, current_error, train_accuracy)) 
		print('epoch: {} - error: {} - thang (train accuracy?): {}'.format(epoch, error, thang)) 
	out = session.run([WEIGHTS, BIASES, error])

	# access graph via tensorboard --logdir ./graph
	file_writer = tf.summary.FileWriter(PATH['graph'], session.graph)

	# close the session
	session.close()

	# print results
	print(out)

def flow():
	build_and_run_graph()

def main():
	enforce_python_version()
	flow()

if __name__ == '__main__':
	main()
