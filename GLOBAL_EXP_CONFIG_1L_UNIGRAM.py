from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os.path
from ENVIRONMENT import *

environment = Environment()
base_path = environment.base_path
ds_path = environment.dataset_base_path

GLOBAL = {
	'numpy_seed': 666,
	'log_format': '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s',
	'log_dir': base_path + '/logs/1layer/unigram/',
	'reports_dir': base_path + '/reports/1layer/unigram/',
	'fullds_reports_dir': base_path + '/reports/1layer/unigram/fullds/',
	'tensorflow_dir': base_path + '/tensorflow/1layer/unigram/',
	'checkpoints_dir':base_path + '/checkpoints/1layer/unigram/',
	'executed_path':base_path + '/executed/1layer/unigram/',
	'data_dir': ds_path + '/malware_selected_1gram_mini.pkl',
	'fullds_data_dir':ds_path + '/malware_selected_1gram.pkl',
	

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
		#'activation' : 'sigmoid',
		'activation' : 'softmax',
		'loss_function' : 'categorical_crossentropy',
		'optimizer' : SGD(lr=0.01),
		'use_last_dim_as_classifier' : False,
		'classifier_dim' : 9
	}


}


MAP_DIMS = {
	'AE_UNIGRAMA_1L_UNDER_F0_1': [96,9],
	'AE_UNIGRAMA_1L_UNDER_F0_2': [96,19],
	'AE_UNIGRAMA_1L_UNDER_F0_3': [96,28],
	'AE_UNIGRAMA_1L_UNDER_F0_4': [96,38],
	'AE_UNIGRAMA_1L_UNDER_F0_5': [96,48],
	'AE_UNIGRAMA_1L_UNDER_F0_6': [96,57],
	'AE_UNIGRAMA_1L_UNDER_F0_7': [96,67],
	'AE_UNIGRAMA_1L_UNDER_F0_8': [96,76],
	'AE_UNIGRAMA_1L_UNDER_F0_9': [96,86],
	'AE_UNIGRAMA_1L_OVER_F1_0' : [96,96],
	'AE_UNIGRAMA_1L_OVER_F1_1' : [96,105],
	'AE_UNIGRAMA_1L_OVER_F1_2' : [96,115],
	'AE_UNIGRAMA_1L_OVER_F1_3' : [96,124],
	'AE_UNIGRAMA_1L_OVER_F1_4' : [96,134],
	'AE_UNIGRAMA_1L_OVER_F1_5' : [96,144],
	'AE_UNIGRAMA_1L_OVER_F1_6' : [96,153],
	'AE_UNIGRAMA_1L_OVER_F1_7' : [96,163],
	'AE_UNIGRAMA_1L_OVER_F1_8' : [96,172],
	'AE_UNIGRAMA_1L_OVER_F1_9' : [96,182],
	'AE_UNIGRAMA_1L_OVER_F2_0' : [96,192],
}

def get_ae_callbacks(network_name):
	ae_callbacks = [
		EarlyStopping(monitor='val_loss', min_delta=0.01, patience=100, verbose=1, mode='min'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name , histogram_freq=1, write_graph=True)	
	]
	return ae_callbacks

def get_mlp_callbacks(network_name):

	mlp_callbacks = [
		EarlyStopping(monitor='acc', min_delta=0.01, patience=100, verbose=1, mode='max'),
		ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '_mlp.h5', monitor='val_acc', save_best_only=True, verbose=1), 
		TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name + '_mlp', histogram_freq=1, write_graph=True)	
	]
	return mlp_callbacks

