from __future__ import division, print_function, absolute_import

import CNN as preprocess

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf

# Building convolutional network
network = input_data(
	shape=(None, preprocess.input_shape[0], preprocess.input_shape[1]),
	name='input')

network = tf.Print(network, data=[network])

print("------------------------------------------------------------------------")
print(network)

network = tf.Print(network, data=[network])

filter_sizes = [32, 16, 8, 4, 3, 2]

conveds = list()
pooleds = list()
for filter_size in filter_sizes:
	conved = tflearn.conv_1d(network, 5, filter_size,
		name="{}conv1d".format(filter_size), padding="same",
		activation='leaky_relu', regularizer="L2")
	conveds.append(conved)
	pooled = tflearn.max_pool_1d(conved, filter_size, 64,
		name="{}maxpool1d".format(filter_size))
	pooleds.append(pooled)

fc_in = tflearn.merge(pooleds, mode="concat", axis=1)

fc_in = tflearn.flatten(fc_in)

network = tflearn.fully_connected(fc_in, 128, activation='leaky_relu')
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, 64, activation='leaky_relu')
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, 2, activation='softmax')
print(network)
network = tflearn.regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

import h5py
h5f = h5py.File("data/database.hdf5", 'r')
X = h5f['text']
# 12  4 minutes into the future
Y = h5f["prices_up"][12]
Y = tflearn.one_hot_encoding(Y, 2)

print(X)
print(Y)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)


model.fit({'input': X}, {'target': Y},
           validation_batch_size=0.01, shuffle=True, batch_size=1024,
           snapshot_step=100, show_metric=True, run_id='sentimentCNN')

