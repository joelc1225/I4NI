#!/usr/bin/env python

from os import path

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

PATH = dict()
PATH['repo'] = path.abspath(path.dirname(__file__))
PATH['graph'] = path.join(PATH['repo'], 'graph')
print(PATH['graph'])

def enforce_python_version():
	if sys.version_info[0] != 2 or sys.version_info[1] != 7:
		raise SystemExit('Please use Python version 2.7.')

def read_dataset():
	df = pd.read_csv("datasets/Sonar.csv")
	print(len(df.columns))
	# features (input)
	feature = df[df.columns[0:60]].values
	# labels (desired output)
	y = df[df.columns[60]]

	encoder = LabelEncoder()
	encoder.fit(y)
	y = encoder.transform(y)
	Y = one_hot_encode(y)
	return (feature, Y)

def one_hot_encode(labels):
	num_labels = len(labels)
	num_unique_labels = len(np.unique(labels))
	encode = np.zeros((num_labels, num_unique_labels))
	encode[np.arange(num_labels), labels] = 1
	return encode

def build_graph():
	# build a graph
	# single nodes
	# node1 = tf.constant(3.0, tf.float32)
	node1 = tf.constant(3.0, tf.float64)
	node2 = tf.constant(4.0, tf.float64)
	node3 = node1 * node2
	print(node1)
	print(node2)
	# session.run([node1, node2, node3])

	a = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)
	c = a * b
	# out = session.run(c, {a: [1, 2],b: [2, 4],})

	# desired W is 6
	W = tf.Variable(3.0)
	# desired b is 2
	b = tf.Variable(1.0)
	x = tf.constant([1.0, 2.0, 3.0])
	# gives result of 4 7 10
	result = W * x + b
	desired_result = tf.constant([8.0, 14.0, 20.0])
	difference = result - desired_result
	# 165
	# desired error is 0
	error = tf.reduce_sum(tf.square(difference))
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	thang = optimizer.minimize(error)

	# VIDEO EXAMPLES START HERE
	feature, Y = read_dataset()
	feature, Y = shuffle(feature, Y, random_state=1)
	train_x, test_x, train_y, test_y = train_test_split(
			feature,
			Y,
			test_size=0.2,
			random_state=415)
	print('shapes are:')
	print(type(feature))
	print(feature.shape)
	print(train_x.shape)
	print(train_y.shape)
	print(test_x.shape)

	learning_rate = 0.3
	training_epochs = 1000
	error_history = np.empty(shape=[1], dtype=float)
	num_dim = feature.shape[1]
	print("num_dim: ", num_dim)
	num_categories = 2
	# number neurons for each layer
	num_neurons = [num_dim, 60, 60, 60, 60, num_categories]

	x = tf.placeholder(tf.float32, [None, num_dim])
	W = tf.Variable(tf.zeros([num_dim, num_categories]))
	b = tf.Variable(tf.zeros([num_categories]))
	y_ = tf.placeholder(tf.float32, [None, num_categories])

weights = {
	'h1': tf.Variable(tf.truncated_normal([num_neurons[0], num_neurons[1])),
	'h2': tf.Variable(tf.truncated_normal([num_neurons[1], num_neurons[2])),
	'h3': tf.Variable(tf.truncated_normal([num_neurons[2], num_neurons[3])),
	'h4': tf.Variable(tf.truncated_normal([num_neurons[3], num_neurons[4])),
	'out': tf.Variable(tf.truncated_normal([num_neurons[4], num_neurons[5])),
}
biases: {
	'b1': tf.Variable(tf.truncated_normal([num_neurons[1]])),
	'b2': tf.Variable(tf.truncated_normal([num_neurons[2]])),
	'b3': tf.Variable(tf.truncated_normal([num_neurons[3]])),
	'b4': tf.Variable(tf.truncated_normal([num_neurons[4]])),
	'out': tf.Variable(tf.truncated_normal([num_neurons[5]])),
}

def multilayer_perceptron(x, weights, biases):
	layers = []
	# input layer (here x = feature)
	layers[0] = None
	# hidden layer with sigmoid
	layers[1] = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layers[1] = tf.nn.sigmoid(layer[1])
	# hidden layer with sigmoid activation
	layers[2] = tf.add(tf.matmul(x, weights['h2']), biases['b2'])
	layers[2] = tf.nn.sigmoid(layer[2])
	# hidden layer with sigmoid activation
	layers[3] = tf.add(tf.matmul(x, weights['h3']), biases['b3'])
	layers[3] = tf.nn.sigmoid(layer[3])
	# hidden layer with RELU activation
	layers[4] = tf.add(tf.matmul(x, weights['h4']), biases['b4'])
	layers[4] = tf.nn.relu(layer[4])
	# output later
	layers[5] = tf.add(tf.matmul(x, weights['out']), biases['out'])
	return layers[5]


def run_graph():
	# run a graph
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	session = tf.Session()
	session.run(init)
	# was 'model_path' in demo
	saver.restore(session, PATH['repo'])
	for _ in range(999):
		session.run(thang)
	out = session.run([W, b, error])

	# access graph via tensorboard --logdir ./graph
	file_writer = tf.summary.FileWriter(PATH['graph'], session.graph)

	# close the session
	session.close()

	# print results
	print(out)

def flow():
	build_graph()
	run_graph()

def main():
	enforce_python_version()
	flow()

if __name__ == '__main__':
	build_graph()
