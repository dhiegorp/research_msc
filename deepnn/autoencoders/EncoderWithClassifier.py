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

class EncoderWithClassifier:

	def __init__(self, encoder_model, name='', activation='sigmoid', loss_function='categorical_crossentropy', optimizer=SGD(lr=0.01), use_last_dim_as_classifier_dim=True, classifier_dim=None):
		self.__encoder_model = encoder_model
		self.__name = name
		self.__activation = activation
		self.__loss_function = loss_function
		self.__optimizer = optimizer
		self.__trained = True
		self.__validated = True
		self.__use_last_dim_as_classifier_dim = use_last_dim_as_classifier_dim
		self.__classifier_dim = classifier_dim
		self.__validateDimensions()
		self.__generateClassifier()
		self.__compile()


	def __validateDimensions(self):
		if not self.__use_last_dim_as_classifier_dim and self.__classifier_dim <= 0:
			raise ValueError("The number of neurons in a layer (classifier_dim) must be greater than zero")

	def __generateClassifier(self):
		ae_output = self.__encoder_model.layers[-1].output
		
		dim = None

		if not self.__use_last_dim_as_classifier_dim:
			dim = self.__classifier_dim
		else:
			dim = self.__encoder_model.layers[-1].units


		self.__classifier_layers = Dense(dim, activation=self.__activation, name='classifier')(ae_output)
		self.__classifier = Model(inputs=[self.__encoder_model.input], outputs=[self.__classifier_layers])

	def __compile(self):
		self.__classifier.compile(loss=self.__loss_function, optimizer=self.__optimizer, metrics=['acc'])



	def __stats(self, path=None):
			
		file_pattern = path + self.__name + '.{0}.{1}'

		classifier_predictions_max = np.argmax(self.__classifier_predictions, axis=1)
		Ymax = np.argmax(self.__eval_label, axis=1)

		self.__confusion_matrix = ConfusionMatrix(Ymax, classifier_predictions_max)

		self.__status_dump(file_pattern, self.__confusion_matrix, html=True, string=True, pickle=True, stats_as_txt=True, latex=True)

		


	def __status_dump(self, file_pattern, confusion_matrix , html=False,string=False, pickle=False, stats_as_txt=False, latex=False):
		dataframe = self.__confusion_matrix.to_dataframe()
		
		if html:
			with open(file_pattern.format('confusion_matrix','html'),'w') as file:
				file.write(dataframe.to_html())
		if string:
				with open(file_pattern.format('confusion_matrix','txt'),'w') as file:
					file.write(dataframe.to_string())
		if pickle:
				dataframe.to_pickle(file_pattern.format('confusion_matrix','pickle'))

		if stats_as_txt:
			with open(file_pattern.format('stats','.txt'),'w') as file:
					file.write(str(confusion_matrix.stats()))
		if latex:
			with open(file_pattern.format('confusion_matrix','.latex_table'),'w') as file:
					file.write(dataframe.to_latex())


	def eval(self, feature=None, label=None):
		self.__eval_feature = feature
		self.__eval_label = label
		self.__classifier_predictions = self.__classifier.predict(feature)

	def train(self,feature=None, label=None, validation=None, epochs=None, batch_size=None, shuffle=True, store_history=True, early_stopping=None, save_every=1, callbacks=None):

		h = self.__classifier.fit(x=feature, y=label, 
			validation_data = validation,
			batch_size=batch_size, epochs=epochs, 
			shuffle=shuffle, callbacks=callbacks)


		if store_history:
			self.__history = h


	def eval_stats(self, reportpath):
		self.__stats(path=reportpath)

	@property
	def classifier(self):
		return self.__classifier

					
