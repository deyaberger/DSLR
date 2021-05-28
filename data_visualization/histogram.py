from histo_course import PlotHistogram
import argparse
import sys

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to test the model', type=str)
	parser.add_argument('-c', '--course', help='name of course to plot', default="Arithmancy", type=str)
	args = parser.parse_args()
	if not args.datafile.endswith(".csv"):
		print("Error: Datafile must be a .csv file")
		sys.exit()
	return (args)

if __name__ == '__main__':
	args = parse_arguments()
	histo = PlotHistogram(args)
	histo.show_histo()