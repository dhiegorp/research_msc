import sys
import os
import csv
import logging
LOG_FORMAT = '[%(asctime)s %(filename)s:%(lineno)s]: %(message)s'
logging.basicConfig(format=LOG_FORMAT, filename='parser.log', level=logging.DEBUG)


base_directory = 'D:/malware_dataset/train'
parse_dump_dir = base_directory + '/parse_dump'
csv_for_target_def = base_directory + '/trainLabels.csv'
file_ext = '.asm'
target_def = {}

def load_filelist():
	global csv_for_target_def, target_def
	with open(csv_for_target_def,'r') as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			target_def[row['Id']] = row['Class']


def filename(base_directory, file_id, file_ext ):
	return '{}/{}{}'.format(base_directory, file_id, file_ext)

def retrieve_filename(file_id):
	global base_directory, file_ext

	return filename(base_directory, file_id, file_ext)


def is_bytearray(line):
	isbytearray = False

	try:
		bytearray.fromhex(line)
		isbytearray = True
	except Exception:
		pass

	return isbytearray	

def parse_line(line):
	'''
		Not for the faint-hearted
	'''
	
	try:
		if len(line) > 0:
			partial = line.replace('+','').replace('?','').replace(';','').replace(':','')
			partial = [token for token in line.rpartition('\t')[:1][0].split()[1:] if len(token) == 2 and token[0].isupper()]
			partial = ' '.join(partial)

			#last test, if its a valid bytearray with hexadecimal bytes

			bytearray.fromhex(partial)

			return partial 

	except (ValueError, Exception):
		pass

	return None

def parse():
	global target_def, base_directory, file_ext
	cnt = 0
	for file_id in target_def:
		target = target_def[file_id]
		filen = retrieve_filename(file_id)
		parsef = filename(parse_dump_dir, file_id + '_' + target, file_ext + '_prs')
		try:
			logging.debug('parsing ' + filen + ' and dumping to ' + parsef)
			with open(filen, 'r', encoding='Latin1') as asmfile:
				with open(parsef, 'a') as prs_file:
					cnt = cnt + 1
					for line in asmfile:
						str = parse_line(line)

						if not str is None and len(str) > 0:
							prs_file.write(str + '\n')

		except Exception:
			pass	

	logging.debug('parsed {} files '.format(cnt))



def main():
	global target_def
	load_filelist()
	parse()
	
if __name__ == '__main__':
	main()