'''
Most of the code from:
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv1D, GlobalMaxPooling1D, Concatenate, Flatten
from keras import regularizers
import keras.metrics
import os
import h5py

import tensorflow as tf


import csv

import numpy as np

import CNN as preprocess

batch_size = 512

epochs = 100
data_augmentation = True

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_cifar10_trained_model.h5'

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, lines, seek_pos, csv_file_name="data/made.csv",
				 batch_size=batch_size, dim=preprocess.input_shape,
				 n_classes=2, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.csv_file_name = csv_file_name
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.lines = lines

		self.seek_pos = seek_pos

		self.csv_file = open(csv_file_name, "r", newline='')
		self.reader = csv.reader(iter(self.csv_file, ''))

		self.text_pos = 3
		self.training_pos = 16 # 15 minutes into the future

		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.lines)) / self.batch_size)

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of lines
		lines = [self.lines[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(lines)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.lines))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, lines):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, line in enumerate(lines):
			self.csv_file.seek(self.seek_pos[line])
			line_data = next(self.reader)
			# Store sample
			X[i,] = preprocess.vectorize(line_data[self.text_pos])

			# Store class
			y[i] = preprocess.make_label(line_data, self.training_pos)

		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

class SimpleCNN:

	def __init__(self):

		num_classes = 2

		inputs = Input(shape=(
			preprocess.input_shape[0], preprocess.input_shape[1]),
			dtype="float32", name="cnn_input")
		
		filter_sizes = [32, 16, 8, 4, 3, 2]

		conveds = list()
		pooleds = list()
		for filter_size in filter_sizes:
			conved = Conv1D(filters=5, kernel_size=filter_size,
				name="{}conv1d".format(filter_size), padding="same",
				activation=keras.layers.LeakyReLU(alpha=.001),
				kernel_regularizer=regularizers.l2(0.01))(inputs)
			conveds.append(conved)
			pooled = GlobalMaxPooling1D()(conved)
			pooleds.append(pooled)

		fc_in = Concatenate(axis=1)(pooleds)

		network = Dense(128)(fc_in)
		network = keras.layers.LeakyReLU(alpha=.001)(network)
		network = Dropout(0.5)(network)
		network = Dense(64)(fc_in)
		network = keras.layers.LeakyReLU(alpha=.001)(network)
		network = Dropout(0.5)(network)
		network = Dense(num_classes)(fc_in)
		outputs = Activation('softmax')(network)

		model = keras.models.Model(inputs=inputs, outputs=outputs)

		# initiate Adam optimizer
		opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
			epsilon=None, decay=0.0, amsgrad=False)

		save_dir = os.path.join(os.getcwd(), 'saved_models')
		model_path = os.path.join(save_dir, model_name)
		self.checkpoint_filepath=os.path.join(
			save_dir, "logs/weights.best.hdf5")

		checkpoint_dir = os.path.dirname(self.checkpoint_filepath)
		if not os.path.isdir(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		if os.path.isfile(self.checkpoint_filepath):
			model.load_weights(self.checkpoint_filepath)
		elif os.path.isfile(model_path):
			model.load_weights(model_path)

		# Let's train the model using RMSprop
		model.compile(loss='categorical_crossentropy',
					optimizer=opt,
					metrics=['accuracy', keras.metrics.categorical_accuracy])

		self.model = model

	def train(self):

		print('Using real-time data augmentation.')

		csv_file_name = "data/made.csv"
		count = 0
		with open(csv_file_name, "r") as f:
			for line in f:
				count += 1
		
		import progressbar
		widgets=[progressbar.Bar(),
			' [', progressbar.Timer(), '] ',
			progressbar.Bar(),
			' (', progressbar.ETA(), ') ', progressbar.AnimatedMarker()]

		seek_pos = [0] * count
		with open(csv_file_name, "r") as f:
			seek_pos[0] = f.tell()
			for i in progressbar.progressbar(range(1, count), max_value=count):
				f.readline()
				val = f.tell()
				
				try:
					seek_pos[i] = val
				except OverflowError:
					print(val)
					raise

		indexes = np.arange(count)
		np.random.shuffle(indexes)

		training = indexes[: int(count * 0.9)]
		testing = indexes[int(count * 0.9) : ]

		# This will do preprocessing and realtime data augmentation:
		training_datagen = DataGenerator(lines=training, seek_pos=seek_pos)
		testing_datagen = DataGenerator(lines=testing, seek_pos=seek_pos)

		tbCallBack = keras.callbacks.TensorBoard(write_grads=True)
		ckpntCallBack = keras.callbacks.ModelCheckpoint(
			self.checkpoint_filepath, monitor='val_acc', verbose=1,
			save_best_only=True, mode='max', period=25)

		# Fit the model on the batches generated by datagen.flow().
		self.model.fit_generator(training_datagen,
							epochs=epochs,
							validation_data=testing_datagen,
							workers=4,
							callbacks=[ckpntCallBack, tbCallBack])

		# Save model and weights
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		self.model.save(model_path)
		print('Saved trained model at %s ' % model_path)

		testing.on_epoch_end()
		x, y = testing[0]
		# Score trained model.
		scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
		print('Test loss:', scores[0])
		print('Test accuracy:', scores[1])
		print("Categorical Accuracy", scores[2])

	def predict(self, *args, **kw):
		return self.model.predict(*args, **kw)

	def predict_image(self, image):
		import numpy as np
		import cv2
		keras_in = np.expand_dims(image, 0)
		keras_in = np.expand_dims(keras_in, len(keras_in.shape))

		predictions = self.predict(keras_in)

		for i in range(predictions.shape[1]):
			print("{} : {}".format(predictions[0, i], self.categories_list[i]))
		
		font = cv2.FONT_HERSHEY_SIMPLEX
		cat = self.categories_list[np.argmax(predictions)]
		cv2.putText(image,cat,(10,60), font,1,(255,255,255),2,cv2.LINE_AA)

		cv2.imshow('image',image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	simpleCNN = SimpleCNN()
	simpleCNN.train()
