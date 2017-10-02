import os.path


def mark_as_done(network_name_path):
	with open(network_name_path, 'a') as file:
		file.write('done!');

def is_executed(network_name_path):
	return os.path.isfile(network_name_path)

def extract_name(str):
	return str[0].split('.')[0]

