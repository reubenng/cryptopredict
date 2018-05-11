from __future__ import division, print_function, absolute_import

import preprocess

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(
	shape=[None, preprocess.input_shape[0], preprocess.input_shape[1]],
	name='input')

filter_sizes = [32, 16, 8, 4, 3, 2]

conveds = list()
pooleds = list()
for filter_size in filter_sizes:
	conved = conv_1d(network, 5, filter_size,
		name="{}conv1d".format(filter_size), padding="same",
		activation='relu', regularizer="L2")
	conveds.append(conved)
	pooled = max_pool_1d(conved, kernel_size, 64,
		name="{}maxpool1d".format(filter_size))
	pooleds.append(pooled)

fc_in = tflearn.merge(pooleds, mode="concat", axis=1)

fc_in = tflearn.flatten(fc_in)

network = fully_connected(fc_in, 128, activation='leaky_relu')
network = dropout(network, 0.8)
network = fully_connected(network, 64, activation='leaky_relu')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

import h5py
h5f = h5py.File("data/database.hdf5", 'r')
X = h5f['text']
# 12  4 minutes into the future
Y = h5f["prices_up"][12]
Y = one_hot_encoding(Y, 2)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)


model.fit({'input': X}, {'target': Y},
           validation_batch_size=0.01), shuffle=True, batch_size=1024,
           snapshot_step=100, show_metric=True, run_id='sentimentCNN')
