from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os.path

GLOBAL = {
	'numpy_seed': 666,
	'log_format': '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s',
	'log_dir': 'E:/research/research_msc/logs/onelayer/bigram/',
	'reports_dir':'E:/research/research_msc/reports/onelayer/bigram/',
	'tensorflow_dir':'E:/research/research_msc/tensorflow/onelayer/bigram/',
	'checkpoints_dir':'E:/research/research_msc/checkpoints/onelayer/bigram/',
	'executed_path':'E:/research/research_msc/executed/onelayer/bigram/',
	'executed_dir':'E:/research/research_msc/executed/onelayer/bigram/',
	'data_dir':'E:/research/malware_dataset/malware_selected_2gram_mini.pkl',
	#'log_dir': 'c:/users/dhieg/research/research_msc/logs/onelayer/bigram/',
	#'reports_dir':'c:/users/dhieg/research/research_msc/reports/onelayer/bigram/',
	#'tensorflow_dir':'c:/users/dhieg/research/research_msc/tensorflow/onelayer/bigram/',
	#'checkpoints_dir':'c:/users/dhieg/research/research_msc/checkpoints/onelayer/bigram/',
	#'executed_dir':'c:/users/dhieg/research/research_msc/executed/onelayer/bigram/',
	#'data_dir':'c:/users/dhieg/research/malware_dataset/malware_selected_2gram_mini.pkl',
	
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

def get_ae_callbacks(network_name):
	ae_callbacks = [
		EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='min'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name , histogram_freq=1, write_graph=True)	
	]
	return ae_callbacks

def get_mlp_callbacks(network_name):

	mlp_callbacks = [
		EarlyStopping(monitor='acc', min_delta=0.01, patience=50, verbose=1, mode='max'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '_mlp.h5', monitor='val_acc', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name + '_mlp', histogram_freq=1, write_graph=True)	
	]
	return mlp_callbacks