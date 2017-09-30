import numpy as np
from sklearn.externals import joblib
from keras.utils import to_categorical
from sklearn.preprocessing import normalize


class DatasetLoader:
	def __init__(self, filepath, targets_list=[], maintain_originals=False, normalize=False, one_hot_encoding=True):
		self.__filepath = filepath
		self.__targets = targets_list
		self.__state = 0
		self.__maintain_originals = maintain_originals
		self.__normalize = normalize
		self.__one_hot = one_hot_encoding
		self.__split_class_from_data()
		self.__normalize_and_fit()



	def __split_class_from_data(self):
		if not self.__filepath:
			raise ValueError('filepath not found. Impossible to load')

		train_full, validation_full = joblib.load(self.__filepath)

		self.__state = 1

		self.__train_features = train_full[:,:-1]
		train_y = train_full[:,[-1]]

		self.__validation_features = validation_full[:,:-1]
		validation_y = validation_full[:,[-1]]
		

		if self.__one_hot:
			self.__train_targets = to_categorical( train_y )
			self.__validation_targets = to_categorical(validation_y)
		else:
			self.__train_targets = train_y
			self.__validation_targets = validation_y

		

		if self.__maintain_originals:
			self.__original_train_x = self.__train_features
			self.__original_train_y = train_y
			self.__original_validation_x = self.__validation_features
			self.__original_validation_y = validation_y

	def recover_originals(self):
		if self.__state == 0:
			raise ValueError('No data was loaded!')
		if self.__maintain_originals == False:
			raise ValueError('maintain_originals was set to False, so no originals were stored!')
		return self.__original_train_x, self.__original_train_y,  self.__original_validation_x, self.__original_validation_y

	def __adjust(self, targets ):
		if self.__one_hot:
			return targets[:,1:]
		return targets

	def __normalize_and_fit(self):
		if self.__normalize:
			self.__norm_train_features = normalize(self.__train_features)
			self.__norm_validation_features = normalize(self.__validation_features)
			self.__norm_train_targets = self.__adjust(self.__train_targets)
			self.__norm_validation_targets = self.__adjust(self.__validation_targets)

	def __call__(self):
		if self.__normalize:
			return self.__norm_train_features, self.__norm_train_targets, self.__norm_validation_features, self.__norm_validation_targets
		else:
	 		return self.__train_features, self.__train_targets, self.__validation_features, self.__validation_targets
	 




