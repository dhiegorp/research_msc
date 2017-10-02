from dataset_loader import *
import math
from sklearn.externals import joblib

class SampleSplitter:
	def __init__(self, dataset, factor, train_factor, target_list, dump_to=None, reshuffle=False):
			self.__dataset = dataset
			self.__factor = factor
			self.__train_factor = train_factor
			self.__reshuffle = reshuffle
			self.__target_list = target_list
			self.__dump_to= dump_to
			self.__validate()

	def __validate(self):
		if self.__dataset == None:
			raise ValueError('the path for dataset was not configured!')

		if self.__factor == None:
			raise ValueError('a factor is required for slicing the dataset')

		if self.__target_list == None:
			raise ValueError('a target list must be provided')

	def __load(self):

		ds_loader = DatasetLoader(self.__dataset, targets_list=self.__target_list, normalize=True, maintain_originals=True, one_hot_encoding=False)
		self.__Xt, self.__yt, self.__Xv, self.__yv = ds_loader()
		msg = """
	=======================================
	loading malware dataset on = {}	
	trainx shape = {}
	trainy shape = {}
	valx shape = {}
	valy shape = {}
	=======================================
			""".format(self.__dataset, str(self.__Xt.shape), str(self.__yt.shape), str(self.__Xv.shape), str(self.__yv.shape))
		print(msg)

	def __gen_counter_obj(self):
		cnt = {}

		for c in self.__target_list:
			cnt[c] = 0

		return cnt

	def __count(self):
		
		cnt = self.__gen_counter_obj()

		for i in self.__yt:
			for c,v in cnt.items():
				if c == i:
					cnt[c] = v + 1

		for i in self.__yv:
			for c,v in cnt.items():
				if c == i:
					cnt[c] = v + 1		

		cc = {}	

		

		for k,v in cnt.items():
			final_num = math.ceil(v * self.__factor)
			train_factor = math.ceil(final_num * self.__train_factor)
			cc[k] = {'total': v, 'train': train_factor, 'validation': math.fabs(final_num - train_factor) }

		return cc

	def __get_samples(self, counter):
		xval = []
		yval = []
		xtra = []
		ytra = []


		for k,v in counter.items():
			print('getting ', v['train'], ' samples for class ', k)
			acc = 1
			for num, row in enumerate(self.__Xt):
				if self.__yt[num] == k and acc < v['train']:
					xtra.append(row)
					ytra.append(self.__yt[num])
					acc = acc + 1

			acc = 1
			for num, row in enumerate(self.__Xv):
				if self.__yv[num] == k and acc < v['validation']:
					xval.append(row)
					yval.append(self.__yv[num])
					acc = acc + 1
			

		self.__xtfinal = np.array(xtra)
		self.__ytfinal = np.array(ytra)
		self.__xvfinal = np.array(xval)
		self.__yvfinal = np.array(yval)

		print('xtfinal> ', self.__xtfinal.shape)
		print('ytfinal> ', self.__ytfinal.shape)
		print('xvfinal> ', self.__xvfinal.shape)
		print('yvfinal> ', self.__yvfinal.shape)

		ret = ( np.c_[self.__xtfinal, self.__ytfinal] , np.c_[self.__xvfinal, self.__yvfinal])
		
		joblib.dump(ret, self.__dump_to)
 


	def process(self):
		self.__load()
		count = self.__count()
		print(count)		
		self.__get_samples(count)



def main():


	process_list = [
		('e:/research/malware_dataset/malware_selected_1gram.pkl', 'e:/research/malware_dataset/malware_selected_1gram_mini.pkl'),
		('e:/research/malware_dataset/malware_selected_2gram.pkl', 'e:/research/malware_dataset/malware_selected_2gram_mini.pkl'),
		('e:/research/malware_dataset/malware_selected_3gram.pkl', 'e:/research/malware_dataset/malware_selected_3gram_mini.pkl')
	]	

	TOTAL_SLICE = 0.25
	TRAIN_SPLIT = 0.6
	CLASS_LIST = [1,2,3,4,5,6,7,8,9]
	RESHUFFLE = False


	for item in process_list:
		ss = SampleSplitter(item[0], factor=TOTAL_SLICE, train_factor=TRAIN_SPLIT, target_list = CLASS_LIST,  dump_to=item[1], reshuffle=RESHUFFLE)
		ss.process()




if __name__ == '__main__':
	main()