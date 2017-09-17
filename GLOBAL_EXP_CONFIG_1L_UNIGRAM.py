from keras.optimizers import SGD
import os.path

GLOBAL = {
	'numpy_seed': 666,
	'log_format': '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s',
	'log_dir': 'C:/Users/dhieg/research/research_msc/logs/onelayer/unigram/',
	'reports_dir':'C:/Users/dhieg/research/research_msc/reports/onelayer/unigram/',
	'tensorflow_dir':'C:/Users/dhieg/research/research_msc/tensorflow/onelayer/unigram/',
	'checkpoints_dir':'C:/Users/dhieg/research/research_msc/checkpoints/onelayer/unigram/',
	'executed_dir':'C:/Users/dhieg/research/research_msc/executed/onelayer/unigram/',
	'data_dir':'C:/Users/dhieg/research/malware_dataset/malware_selected_1gram.pkl',
	'data_target_list' : [1,2,3,4,5,6,7,8,9],
	'epochs': 1000,
	'batch': 32,
	'store_history' : True,
	'shuffle_batches' : True,
	'autoencoder_configs' : {
		'hidden_layer_activation' : 'relu',
		'output_layer_activation' : 'relu',
		'loss_function' : 'mse',
		'optimizer': SGD(lr=0.01),
		'discard_decoder_function': True
	},
	'mlp_configs': {
		'activation' : 'sigmoid',
		'loss_function' : 'categorical_crossentropy',
		'optimizer' : SGD(lr=0.01),
		'use_last_dim_as_classifier' : False,
		'classifier_dim' : 9
	}



}

def mark_as_done(network_name):
	with open(GLOBAL['executed_dir'] + network_name, 'a') as file:
		file.write('done!');

def is_executed(network_name):
	return os.path.isfile(GLOBAL['executed_dir'] + network_name)

def extract_name(str):
	return str[0].split('.')[0]

