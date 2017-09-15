import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

np.random.seed(666)

def dump(obj, filename=None):
	joblib.dump(obj, filename)

def load(filename=None):
	return joblib.load(filename)

class Ngrams:
	def __init__(self, base_path=None, file_ext=None, n=1):
		self.__base_path = base_path
		self.__file_ext = file_ext
		self.__n = n


	def __generate_fileslist(self):
		ext = self.__file_ext
		path = self.__base_path
		
		files = []
		filenames = []
		
		for f in os.listdir(path):
			if f.endswith(ext):
				files.append( '{}/{}'.format(path, f) )
				filenames.append(f)

		return files, filenames



	def __dump_fileslist(self, filelist=[], filename=None): 

		if filelist is not None and len(filelist) > 0:
			dump(filelist, filename)

	def __load_fileslist(self, file=None):
		if from_file is not None and len(from_file) > 0:
			return load(file)
		return None

	def __TFIDFit(self, filelist=[], max_features=None, smooth=True, norm=None):
		tfidf = TfidfVectorizer(input='filename', ngram_range=(self.__n, self.__n), max_features=max_features, smooth_idf=smooth, norm=norm, dtype=np.float32)
		tfidf.fit(filelist);
		return tfidf

	def __extract_target_matrix(self, files=[]):
		y = [ targets for file in files for targets in file.replace(self.__file_ext,'').split('_')[1] ]
		y = np.matrix( [y], dtype=np.float32 )
		y = y.T
		return y

	def __concat_matrices(self, filelist=[], targets=None, vectorizer=None):
		X = vectorizer.transform(filelist).todense()
		print('X ', X.shape, ' Y ', targets.shape)
		Xy = np.c_[ X, targets ]
		return Xy

	def execute(self):
		filepaths, filenames = self.__generate_fileslist()
		tfidf = self.__TFIDFit(filelist=filepaths)
		y = self.__extract_target_matrix(files=filenames)
		Xy = self.__concat_matrices(filelist=filepaths, targets=y, vectorizer=tfidf)

		filename_pattern = '{}_{}gram.pkl'
		dump( filenames, filename_pattern.format('filename', str(self.__n) ) )
		dump( tfidf, filename_pattern.format('tfidf', str(self.__n) ) )
		dump( tfidf.get_feature_names(), filename_pattern.format('feature_terms', str(self.__n) ) )
		dump( y, filename_pattern.format('target_matrix', str(self.__n) ) )
		dump( Xy, filename_pattern.format('features_and_targets', str(self.__n) ) )














	

	
	
