import sys
import subprocess
import os
import glob
import logging

processes = []
running = []
search_term = None

logFormat = logging.Formatter('[%(asctime)s (line %(lineno)s)]: %(message)s')

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileLogger = logging.FileHandler('launcher.log')
fileLogger.setFormatter(logFormat)

consoleLogger = logging.StreamHandler()
consoleLogger.setFormatter(logFormat)

rootLogger.addHandler(fileLogger)
rootLogger.addHandler(consoleLogger)

logging.debug('sys.argv: ' + str(sys.argv) )

if len(sys.argv) == 2:
	search_term = sys.argv[1]

logging.debug("Starting process launcher with search term \'" + search_term + "\'" )

def load():
	global processes

	filelist = glob.glob( search_term )
	
	logging.debug(str(len(filelist)) + ' item(s) found for search_term ' + search_term) 

	for filename in glob.glob( search_term ):
			logging.debug('Registering ' + filename + ' for parallel execution.')
			processes.append( filename )
	


def run():
	global processes, running

	for proc in processes:
		logging.debug('Opening process for ' + proc)
		subprocess.call([sys.executable, proc])
		#running.append( subprocess.Popen( [sys.executable, proc] ) )


	#for proc in running:
	#	proc.wait()	

def main():
	load()
	run()


if __name__ == '__main__':
	main()
