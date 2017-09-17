import sys
import subprocess


processes = []
running = []

def load():
	global processes

	processes.append('AE_UNIGRAMA_1L_OVER_F1_0.py')
	


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
