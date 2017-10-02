import os.path
import sys
import logging
from GLOBAL_EXP_FUNCTIONS import *
from GLOBAL_EXP_CONFIG_1L_TRIGRAM import *
import numpy as np
from deepnn.autoencoders.Autoencoder import Autoencoder
from datasets.dataset_loader import DatasetLoader
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

network_name = extract_name(sys.argv)
network_name_path = GLOBAL['executed_path'] + network_name

"""
SET ENCODER FUNCTION'S LAYERS ON layers LIST
"""
layers = [10000,3000]

logging.basicConfig(format=GLOBAL['log_format'], filename= GLOBAL['log_dir'] + network_name + '.log', level=logging.DEBUG)

load_ds = None
ae_model = None
mlp_model = None
classifier_predictions = None
trainx, trainy, valx, valy = None, None, None, None

ae_callbacks = [
	EarlyStopping(monitor='val_loss', min_delta=0.01, patience=100, verbose=1, mode='min'),
	ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1), 
	TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name , histogram_freq=1, write_graph=True)	
]

mlp_callbacks = [
	EarlyStopping(monitor='acc', min_delta=0.01, patience=100, verbose=1, mode='max'),
	ModelCheckpoint(GLOBAL['checkpoints_dir'] + network_name + '_mlp.h5', monitor='val_acc', save_best_only=True, verbose=1), 
	TensorBoard(log_dir=GLOBAL['tensorflow_dir'] + network_name + '_mlp', histogram_freq=1, write_graph=True)	
]

def header_log():
	header = """
	=======================================
	network_name = {}
	layers = {}
	using GLOBAL obj = 
		{}
	=======================================
	"""
	logging.debug(header.format(network_name, ','.join(str(layer) for layer in layers),  str(GLOBAL)))
	

def data_init():
	global trainx, trainy, valx, valy, load_ds
	load_ds = DatasetLoader(GLOBAL['data_dir'], targets_list=GLOBAL['data_target_list'], normalize=True, maintain_originals=True)
	trainx, trainy, valx, valy = load_ds()
	msg = """
	=======================================
	loading malware dataset on = {}	
	trainx shape = {}
	trainy shape = {}
	valx shape = {}
	valy shape = {}
	=======================================
	""".format(GLOBAL['data_dir'], str(trainx.shape), str(trainy.shape), str(valx.shape), str(valy.shape))
	logging.debug(msg)
	
def execute_autoencoder():
	global ae_model

	logging.debug("=======================================")
	

	CONFIG = GLOBAL['autoencoder_configs']
	
	logging.debug("setting configurations for autoencoder: \n\t " + str(CONFIG) )
	
	ae_model = Autoencoder(
		layers, 
		name = network_name, 
		hidden_layer_activation = CONFIG['hidden_layer_activation'], 
		output_layer_activation = CONFIG['output_layer_activation'], 
		loss_function = CONFIG['loss_function'], 
		optimizer = CONFIG['optimizer'], 
		discard_decoder_model = CONFIG['discard_decoder_function'])

	logging.debug("training and evaluate autoencoder")	

	ae_model.train_and_eval(
		feature=trainx, 
		feature_validation=valx, 
		epochs=GLOBAL['epochs'], 
		batch_size=GLOBAL['batch'], 
		shuffle=GLOBAL['shuffle_batches'], 
		store_history=GLOBAL['store_history'], 
		callbacks = ae_callbacks )

	logging.debug("trained and evaluated!")	
	
	try: 
		logging.debug("Training history: \n" + str(ae_model.training_history.history) )
	except:
		pass

	logging.debug("done!")		

def execute_mlp():
	global mlp_model, classifier_predictions

	logging.debug("=======================================")

	CONFIG = GLOBAL['mlp_configs']

	logging.debug("setting configurations for classifier: \n\t " + str(CONFIG) )

	mlp_model = ae_model.get_classifier( 
		activation = CONFIG['activation'], 
		loss_function = CONFIG['loss_function'], 
		optimizer = CONFIG['optimizer'], 
		use_last_dim_as_classifier_dim = CONFIG['use_last_dim_as_classifier'], 
		classifier_dim = CONFIG['classifier_dim'])

	logging.debug("training ... ")	

	mlp_model.train( 
		feature=trainx, 
		label=trainy,   
		validation=(valx, valy), 
		epochs=GLOBAL['epochs'], 
		batch_size=GLOBAL['batch'], 
		shuffle=GLOBAL['shuffle_batches'], 
		store_history=GLOBAL['store_history'],
		callbacks=mlp_callbacks )

	logging.debug("trained!")	
	
	try: 
		logging.debug("Training history: \n" + str(ae_model.training_history.history) )
	except:
		pass

	logging.debug('evaluating model ... ')
	
	mlp_model.eval(feature=valx, label=valy)

	logging.debug('evaluated! ')

	logging.debug('generating reports ... ')
	mlp_model.eval_stats(GLOBAL['reports_dir'])

	logging.debug('done!')
	

def execute():
	if is_executed(network_name_path):
		logging.debug("The experiment " + network_name + " was already executed!")
	else:
		logging.debug(">> Initializing execution of experiment " + network_name )
		logging.debug(">> Printing header log")
		header_log()
		logging.debug(">> Loading dataset... ")
		data_init()
		logging.debug(">> Executing autoencoder part ... ")
		execute_autoencoder()
		logging.debug(">> Executing classifier part ... ")
		execute_mlp()
		logging.debug(">> experiment " + network_name + " finished!")
		mark_as_done(network_name_path)

def main():
	execute()


if __name__ == '__main__':
	main()
