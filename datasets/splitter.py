import math
import numpy as np

class DatasetSplitter:
	
	def __init__(self, targets_list=[], filenames_list = [], split_factor=.25, XY=None):
		self.__targets_list = targets_list
		self.__filenames_list = filenames_list
		self.__factor = split_factor
		self.__class_counter = self.__create_target_counter()
		self.__XY = XY

	def __create_target_counter(self):
		if self.__targets_list != None:
			class_counter = {}
			for c in self.__targets_list:
				class_counter.update({c : 0})
			return class_counter
		return None

	def __calculate_sample_by_ratio(self):
		
		self.__training_sample_size = {}
		self.__validation_sample_size = {}

		counter = self.__create_target_counter()
		for name in self.__filenames_list:
			for target in self.__targets_list:
				if name.find('_' + target) > -1:
					counter[target] += 1

		for target in self.__targets_list:
			
			validation = math.ceil( counter[target] * self.__factor )
			training = counter[target] - validation
			
			self.__validation_sample_size.update({target : validation})
			self.__training_sample_size.update({target : training})


	def __create_samples_config(self):
		if self.__targets_list != None:
			self.__samples_config = {}
			for c in self.__targets_list:
				self.__samples_config.update( 
					{ 
						c : { 
							'train' : {'current' : 0, 'limit': self.__training_sample_size[c], 'matrix' : []}, 
						    'validation': {'current' : 0, 'limit': self.__validation_sample_size[c], 'matrix': []}
						}
					}
				)

	def __reshape_matrix(self, matrix):
		if len(matrix.shape) == 3:
			matrix = matrix.reshape(matrix.shape[0], matrix.shape[2])
		else:
			raise ValueError('wrong shape size for matrix! expected 3 and found ' + str(len(matrix.shape)) + ', shape=' + str(matrix.shape))			

	def __split_matrices(self):
		
		for target in self.__targets_list:
			
			cfg = self.__samples_config[target]
			
			if cfg :
			
			for row in self.__XY:
				if int(target) == row[:,[-1]][0]:
					if cfg['validation']['current'] < cfg['validation']['limit']:
						x = cfg['validation']['matrix']
						x.append(row)
						cfg['validation']['current'] += 1

					elif cfg['train']['current'] < cfg['train']['limit']:
						x = cfg['train']['matrix']
						x.append(row)
						cfg['train']['current'] += 1
		
		train_dataset = []
		validation_dataset = []

		for k, v in extr_config.items():
			train_dataset += v['train']['matrix']
			validation_dataset += v['validation']['matrix']

		train_dataset = np.array(train_dataset)
		validation_dataset = np.array(validation_dataset)

		self.__reshape_matrix(train_dataset)
		self.__reshape_matrix(validation_dataset)
		np.random.shuffle(train_dataset)
		np.random.shuffle(validation_dataset)

		self.__train_samples = train_dataset
		self.__validation_samples = validation_dataset

	def split(self):
		self.__calculate_sample_by_ratio()
		self.__create_samples_config()
		self.__split_matrices()
		return (self.__train_samples, self.__validation_samples)				

	@property
	def training_samples(self):
		return self.__train_samples

	@property
	def validation_samples(self):
		return self.__validation_samples		

