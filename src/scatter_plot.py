from classes import ScatterPairPlot
import argparse
import sys

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to test the model', type=str)
	parser.add_argument('-x', '--x_course', help='name of course to plot on x axis. Possible courses : Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination\
                            Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying', default="Defense Against the Dark Arts", type=str)
	parser.add_argument('-y', '--y_course', help='name of course to plot on y axis. Possible courses : See x_course.', default="Astronomy", type=str)
	args = parser.parse_args()
	if not args.datafile.endswith(".csv"):
		print("Error: Datafile must be a .csv file")
		sys.exit()
	return (args)

if __name__ == '__main__':
	args = parse_arguments()
	graph = ScatterPairPlot(args)
	graph.scatter_plot()
