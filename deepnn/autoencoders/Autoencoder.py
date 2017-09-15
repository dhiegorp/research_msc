import sys
sys.path.insert(0, r'../../')
import numpy as np
import logging
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from pandas_ml import ConfusionMatrix
from deepnn.autoencoders.EncoderWithClassifier import EncoderWithClassifier

class Autoencoder:

	def __init__( self, 
		encoder_layers, 
		name='', 
		hidden_layer_activation='relu', 
		output_layer_activation='relu', 
		loss_function='mse', 
		optimizer=SGD(lr=0.01), 
		discard_decoder_model=False):

		self.__name = name
		self.__encoder_layers_config = encoder_layers
		self.__discard_decoder_model = discard_decoder_model
		
		self.__hidden_layers_activation = hidden_layer_activation
		self.__output_layer_activation = output_layer_activation
		self.__loss_function = loss_function
		self.__optimizer = optimizer
		self.__trained = False
		self.__validated = False
		self.__generateModels()
		self.__compile()


	def __validateEncoderLayers(self):
		""" validate the encoder layers configured 
			raise value errors if __encoder_layers_config was not setted or if list len is 'le' than one
		"""
		if not self.__encoder_layers_config:
			raise ValueError('A list with the numbers of neurons in each layer is required.')

		if len(self.__encoder_layers_config) <= 1:
			raise ValueError('To generate an autoencoder you have to provide at least 2 layers (two items of a list).')			

	def __generateEncoder(self, input):

		for id, neurons in enumerate(self.__encoder_layers_config[1:]):
			if id == 0:
				self.__encoder_layers = Dense(neurons, activation=self.__hidden_layers_activation, name='enc{}_{}'.format(id, neurons) )(input)
			else:
				self.__encoder_layers = Dense(neurons, activation=self.__hidden_layers_activation, name='enc{}_{}'.format(id, neurons) )(self.__encoder_layers)
		
		self.__encoder_model = Model(inputs=[input], outputs=[self.__encoder_layers])

	def __generateDecoder(self):

		reversed_encoder_layers = self.__encoder_layers_config[:-1]

		for id, neurons in enumerate( reversed(reversed_encoder_layers) ):
			if id == 0:
				self.__decoder_layers = Dense(neurons, activation = self.__hidden_layers_activation, name='dec{}_{}'.format(id, neurons))(self.__encoder_layers)
			else:
				decoder_activation = ''
				
				if id == len( self.__encoder_layers_config[:-1] ) - 1 :
					decoder_activation = self.__output_layer_activation

				else:
					decoder_activation = self.__hidden_layers_activation

				self.__decoder_layers = Dense(neurons, activation = decoder_activation, name='dec{}_{}'.format(id, neurons))(self.__decoder_layers)
		
		

	def __generateModels(self):

		self.__validateEncoderLayers()

		input = Input(shape=(self.__encoder_layers_config[0],))

		self.__generateEncoder(input)
		self.__generateDecoder()

		self.__autoencoder = Model(inputs=[input], outputs=[self.__decoder_layers])

#		if not self.__discard_decoder_model:
#			decoder_input = self.__encoder_model.layers[-1].output
#
#			self.__decoder_model = Model(inputs=[decoder_input], outputs=[self.__autoencoder.layers[-1](decoder_input)])			


	
 
	def __compile(self):
		self.autoencoder.compile(loss=self.__loss_function, optimizer=self.__optimizer)

	
		
	def train_and_eval(self, feature=None, feature_validation=None, epochs=1000, batch_size=32, shuffle=True, store_history=True, callbacks=None):

		validation_data = None

		if not feature_validation == None:
			validation_data = (feature_validation, feature_validation)

		h = self.__autoencoder.fit(x=feature, y=feature, validation_data=validation_data, shuffle=shuffle, 
			epochs=epochs, batch_size=batch_size, callbacks=callbacks )
		
		if store_history:
			self.__history = h

		self.__trained = True
		self.__validated = True

	def __stats(self):
			
		file_pattern = 'reports/' + self.__name + '.{0}.{1}'

		classifier_predictions_max = np.argmax(self.__classifier_predictions, axis=1)
		Ymax = np.argmax(self.__eval_label, axis=1)

		self.__confusion_matrix = ConfusionMatrix(Ymax, classifier_predictions_max)

		self.__status_dump(file_pattern, self.__confusion_matrix, html=True, string=True, pickle=True, stats_as_txt=True, latex=True)

		


	def __status_dump(self, file_pattern, confusion_matrix , html=False,string=False, pickle=False, stats_as_txt=False, latex=False):
		dataframe = self.__confusion_matrix.to_dataframe()
		
		if html:
			with open(file_pattern.format('confusion_matrix','.html'),'w') as file:
				file.write(dataframe.to_html())
		if string:
				with open(file_pattern.format('confusion_matrix','.txt'),'w') as file:
					file.write(dataframe.to_string())
		if pickle:
				dataframe.to_pickle(file_pattern.format('confusion_matrix','.pickle'))

		if stats_as_txt:
			with open(file_pattern.format('stats','.txt'),'w') as file:
					file.write(str(confusion_matrix.stats()))
		if latex:
			with open(file_pattern.format('confusion_matrix','.latex_table'),'w') as file:
					file.write(dataframe.to_latex())


	def get_classifier(self, activation=None, loss_function=None, optimizer=None, use_last_dim_as_classifier_dim=None, classifier_dim=None):
		if self.__trained and self.__validated:
			classifier = EncoderWithClassifier(self.__encoder_model, name= self.__name + '_classifier', activation=activation, loss_function = loss_function, optimizer = optimizer, use_last_dim_as_classifier_dim = use_last_dim_as_classifier_dim, classifier_dim = classifier_dim)			
			return classifier
		else:
			logging.info("impossible to create a classifier. Autoencoder isn't trained or validated!")
		return None			


	def eval(self, feature=None):
		self.__eval_feature = feature
		self.__classifier_predictions = self.__classifier.predict(feature)


	def eval_stats(self):
		self.__stats()


	@property
	def encoder_model(self):
		return self.__encoder_model

	@property
	def decoder_model(self):
		return self.__decoder_model

	@property
	def autoencoder(self):
		return self.__autoencoder

	@property
	def training_history(self):
		return self.__history






