
def mark_as_done(network_name):
	with open(GLOBAL['executed_dir'] + network_name, 'a') as file:
		file.write('done!');

def is_executed(network_name):
	return os.path.isfile(GLOBAL['executed_dir'] + network_name)

def extract_name(str):
	return str[0].split('.')[0]

