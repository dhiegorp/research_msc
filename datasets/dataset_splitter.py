import math
import os
import numpy as np
from sklearn.externals import joblib
import logging
import ngrams


LOG_FORMAT = '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s'
logging.basicConfig(format=LOG_FORMAT, filename='data_splitter.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
	
def check_class_in_name(name, counter):
	for cls in counter:
		if name.find('_'+str(cls)) > -1:
			counter[cls] += 1

def calc_sample_nums(counter, factor):
	sample_config = {}
	info = 'for class {}, samples for: training {}, validation {}'
	for i in counter:
		validation = math.ceil(counter[i] * factor)
		training = counter[i] - validation
		logging.info(info.format(i,training,validation))
		print(info.format(i,training,validation))
		sample_config[i] = { 'train':{'current':0, 'limit':training, 'matrix':[]},
							  'validation':{'current':0, 'limit':validation, 'matrix':[]}
							 }
	return sample_config

 
def main_unigram():
	
	class_counter = {'1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0 }

	logging.info('loading filenames_1gram.pkl')
	filenames = ngrams.load('filenames_1gram.pkl')
	

	logging.info('counting samples per class')
	for name in filenames:
		check_class_in_name(name, class_counter)

	logging.info('files count by class ' + str(class_counter))
	print('files count by class ', class_counter)

	factor = 0.25
	logging.info('splitting factor = ' + str(factor))

	
	extr_config = calc_sample_nums(class_counter, factor)

	logging.info('loading full dataset from featuresmatrix_and_targets_1gram.pkl...')

	XY = ngrams.load('featuresmatrix_and_targets_1gram.pkl')

	#extr_config = { '2': { 'train' : {'current':0, 'limit': trainingc2, 'matrix' :[] }, 
	#					   'validation' : {'current':0, 'limit': validationc2, 'matrix' :[]} }, 
	#				'3': { 'train' : {'current':0, 'limit': trainingc3, 'matrix' :[]}, 
	#					   'validation' : {'current':0, 'limit': validationc3, 'matrix' :[]} }, 
	#				'5': { 'train' : {'current':0, 'limit': trainingc5, 'matrix' :[]}, 
	#					   'validation' : {'current':0, 'limit': validationc5, 'matrix' :[]} } }

	logging.info('starting splitting process...')

	for target in class_counter:
		cfg = extr_config[target]
		if cfg :
			for row in XY:
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

	if len(train_dataset.shape) == 3:
		train_dataset = train_dataset.reshape(train_dataset.shape[0], train_dataset.shape[2])
	else:
		logging.info('wrong shape size for training dataset! expected 3 and found ' + str(len(train_dataset.shape)) + ', shape=' + str(train_dataset.shape))

	if len(validation_dataset.shape) == 3:
		validation_dataset = validation_dataset.reshape(validation_dataset.shape[0], validation_dataset.shape[2])
	else:
		logging.info('wrong shape size for validation_dataset dataset! expected 3 and found ' + str(len(validation_dataset.shape)) + ', shape=' + str(validation_dataset.shape))


	print('executing shuffle on datasets')
	logging.info('executing shuffle on datasets')
	np.random.shuffle(train_dataset)
	np.random.shuffle(validation_dataset)

	logging.info('training final shape = ' + str(train_dataset.shape))
	logging.info('validation final shape = ' + str(validation_dataset.shape))

	logging.info('dumping to malware_selected_1gram.pkl')

	ngrams.dump( (train_dataset, validation_dataset), 'malware_selected_1gram.pkl')

	logging.info('the end!')

def main_bigram():
	
	class_counter = {'1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0 }

	logging.info('loading filenames_2gram.pkl')
	filenames = ngrams.load('filenames_2gram.pkl')
	

	logging.info('counting samples per class')
	for name in filenames:
		check_class_in_name(name, class_counter)

	logging.info('files count by class ' + str(class_counter))
	print('files count by class ', class_counter)

	factor = 0.25
	logging.info('splitting factor = ' + str(factor))

	
	extr_config = calc_sample_nums(class_counter, factor)

	logging.info('loading full dataset from featuresmatrix_and_targets_2gram.pkl...')

	XY = ngrams.load('featuresmatrix_and_targets_2gram.pkl')

	#extr_config = { '2': { 'train' : {'current':0, 'limit': trainingc2, 'matrix' :[] }, 
	#					   'validation' : {'current':0, 'limit': validationc2, 'matrix' :[]} }, 
	#				'3': { 'train' : {'current':0, 'limit': trainingc3, 'matrix' :[]}, 
	#					   'validation' : {'current':0, 'limit': validationc3, 'matrix' :[]} }, 
	#				'5': { 'train' : {'current':0, 'limit': trainingc5, 'matrix' :[]}, 
	#					   'validation' : {'current':0, 'limit': validationc5, 'matrix' :[]} } }

	logging.info('starting splitting process...')

	for target in class_counter:
		cfg = extr_config[target]
		if cfg :
			for row in XY:
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

	if len(train_dataset.shape) == 3:
		train_dataset = train_dataset.reshape(train_dataset.shape[0], train_dataset.shape[2])
	else:
		logging.info('wrong shape size for training dataset! expected 3 and found ' + str(len(train_dataset.shape)) + ', shape=' + str(train_dataset.shape))

	if len(validation_dataset.shape) == 3:
		validation_dataset = validation_dataset.reshape(validation_dataset.shape[0], validation_dataset.shape[2])
	else:
		logging.info('wrong shape size for validation_dataset dataset! expected 3 and found ' + str(len(validation_dataset.shape)) + ', shape=' + str(validation_dataset.shape))


	print('executing shuffle on datasets')
	logging.info('executing shuffle on datasets')
	np.random.shuffle(train_dataset)
	np.random.shuffle(validation_dataset)

	logging.info('training final shape = ' + str(train_dataset.shape))
	logging.info('validation final shape = ' + str(validation_dataset.shape))

	logging.info('dumping to malware_selected_2gram.pkl')

	ngrams.dump( (train_dataset, validation_dataset), 'malware_selected_2gram.pkl')

	logging.info('the end!')
	
def main_trigram():
	
	class_counter = { '1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0 }

	logging.info('loading filenames_3gram.pkl')
	filenames = ngrams.load('filenames_3gram.pkl')
	

	logging.info('counting samples per class')
	for name in filenames:
		check_class_in_name(name, class_counter)

	logging.info('files count by class ' + str(class_counter))
	print('files count by class ', class_counter)

	factor = 0.25
	logging.info('splitting factor = ' + str(factor))

	
	extr_config = calc_sample_nums(class_counter, factor)

	logging.info('loading full dataset from featuresmatrix_and_targets_3gram.pkl...')

	XY = ngrams.load('featuresmatrix_and_targets_3gram.pkl')

	#extr_config = { '2': { 'train' : {'current':0, 'limit': trainingc2, 'matrix' :[] }, 
	#					   'validation' : {'current':0, 'limit': validationc2, 'matrix' :[]} }, 
	#				'3': { 'train' : {'current':0, 'limit': trainingc3, 'matrix' :[]}, 
	#					   'validation' : {'current':0, 'limit': validationc3, 'matrix' :[]} }, 
	#				'5': { 'train' : {'current':0, 'limit': trainingc5, 'matrix' :[]}, 
	#					   'validation' : {'current':0, 'limit': validationc5, 'matrix' :[]} } }

	logging.info('starting splitting process...')

	for target in class_counter:
		cfg = extr_config[target]
		if cfg :
			for row in XY:
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

	if len(train_dataset.shape) == 3:
		train_dataset = train_dataset.reshape(train_dataset.shape[0], train_dataset.shape[2])
	else:
		logging.info('wrong shape size for training dataset! expected 3 and found ' + str(len(train_dataset.shape)) + ', shape=' + str(train_dataset.shape))

	if len(validation_dataset.shape) == 3:
		validation_dataset = validation_dataset.reshape(validation_dataset.shape[0], validation_dataset.shape[2])
	else:
		logging.info('wrong shape size for validation_dataset dataset! expected 3 and found ' + str(len(validation_dataset.shape)) + ', shape=' + str(validation_dataset.shape))


	print('executing shuffle on datasets')
	logging.info('executing shuffle on datasets')
	np.random.shuffle(train_dataset)
	np.random.shuffle(validation_dataset)

	logging.info('training final shape = ' + str(train_dataset.shape))
	logging.info('validation final shape = ' + str(validation_dataset.shape))

	logging.info('dumping to malware_selected_3gram.pkl')

	ngrams.dump( (train_dataset, validation_dataset), 'malware_selected_3gram.pkl')

	logging.info('the end!')	




if __name__ == '__main__':
	print('exec unigram')
	main_unigram() 
	print('exec bigram')
	main_bigram()
	print('exec trigram')
	main_trigram()