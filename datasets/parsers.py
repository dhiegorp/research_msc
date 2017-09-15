import csv

class ASMOpcodeParser:

	def __init__(self, path=None, filelist_path=None):
		self.__path = path
		self.__filelist_path = filelist_path
		self.__file_extension = 'asm'
		self.__files_list = {}

	def load_fileslist(self):
		with open(self.__filelist_path, 'r') as csv:
			reader = csv.DictReader(csv_file)
			for row in reader:
				target_def[row['Id']] = row['Class']

	def __get_filename(self, file_id, file_extension=None):
		ext = self.__file_extension

		if file_extension:
			ext = file_extension
		
		return '{}/{}{}'.format(self.__path, file_id, ext)


	def __parse_line(self, line):
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

	def parse_files():

		bytearray_file_ext = self.__file_extension + '_prs'

		for file_id, target in self.__files_list.items():
			
			filename = self.__get_filename(file_id)
			bytearray_file_id = file_id + '_' + target
			dump_file = self.__get_filename(bytearray_file_id, file_extension=bytearray_file_ext)

			try:

				with open(filename, 'r', encoding='Latin1') as asm_file:
					with open(dump_file, 'a') as prs_file:
						for line in asm_file:
							parsed_line = parse_line(line)

							if not parse_line is None and len(parse_line) > 0:
								prs_file.write(parse_line + '\n')
			except Exception:
				pass
