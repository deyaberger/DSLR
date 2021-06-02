from classes import PlotHistogram
import argparse
import sys

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to test the model', type=str)
	parser.add_argument('-c', '--course', help='name of course to plot. Possible courses : Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination\
                            Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying', default="Arithmancy", type=str)
	args = parser.parse_args()
	if not args.datafile.endswith(".csv"):
		print("Error: Datafile must be a .csv file")
		sys.exit()
	return (args)

if __name__ == '__main__':
	args = parse_arguments()
	histo = PlotHistogram(args)
	histo.show_histo()