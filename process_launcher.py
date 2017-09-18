import sys
import subprocess
import os
import glob

processes = []
running = []
search_term = "AE_*.py"


print('>>>>>>>>>>>>',len(sys.argv), '-----', sys.argv)
if len(sys.argv) == 2:
	search_term = sys.argv[1]

def load():
	global processes

	filelist = glob.glob( search_term )
	print('For search_term ', search_term, '  ', str(len(filelist)), ' item(s) found') 

	for filename in glob.glob( search_term ):
			print('Registering ', filename, ' for parallel execution.')
			processes.append( filename )
	


def run():
	global processes, running

	for proc in processes:
		running.append( subprocess.Popen( [sys.executable, proc] ) )

	for proc in running:
		proc.wait()	

def main():
	load()
	run()
	#print( extract_name(sys.argv) )


if __name__ == '__main__':
	main()
