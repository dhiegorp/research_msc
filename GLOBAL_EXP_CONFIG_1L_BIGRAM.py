from keras.optimizers import SGD
import os.path

GLOBAL = {
	'numpy_seed': 666,
	'log_format': '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s',
	#'log_dir': 'E:/research/research_msc/logs/onelayer/bigram/',
	#'reports_dir':'E:/research/research_msc/reports/onelayer/bigram/',
	#'tensorflow_dir':'E:/research/research_msc/tensorflow/onelayer/bigram/',
	#'checkpoints_dir':'E:/research/research_msc/checkpoints/onelayer/bigram/',
	#'executed_dir':'E:/research/research_msc/executed/onelayer/bigram/',
	#'data_dir':'E:/research/malware_dataset/malware_selected_2gram.pkl',
	'log_dir': 'c:/users/dhieg/research/research_msc/logs/onelayer/bigram/',
	'reports_dir':'c:/users/dhieg/research/research_msc/reports/onelayer/bigram/',
	'tensorflow_dir':'c:/users/dhieg/research/research_msc/tensorflow/onelayer/bigram/',
	'checkpoints_dir':'c:/users/dhieg/research/research_msc/checkpoints/onelayer/bigram/',
	'executed_dir':'c:/users/dhieg/research/research_msc/executed/onelayer/bigram/',
	'data_dir':'c:/users/dhieg/research/malware_dataset/malware_selected_2gram_mini.pkl',
	
	'data_target_list' : [1,2,3,4,5,6,7,8,9],
	'epochs': 200,
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
