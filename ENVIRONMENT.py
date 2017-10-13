import os

class Environment:
	def __init__(self):
		self.__load_base_path()
		self.__load_dataset_base_path()


	def __load_base_path(self):
		self.__base_path = os.environ['DEEPNN_BASE_PATH']

	def __load_dataset_base_path(self):
		self.__dataset_base_path = os.environ['DEEPNN_DATASET_BASE_PATH'] 

	@property
	def base_path(self):
		return self.__base_path

	@property
	def dataset_base_path(self):
		return self.__dataset_base_path

