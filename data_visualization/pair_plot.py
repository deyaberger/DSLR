import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sys
import argparse
from plot import ScatterPairPlot

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to test the model', type=str)
	parser.add_argument('-f', '--features', help='choose features to display on the pair plot\n\n\
		0: Arithmancy, 1: Astronomy, 2: Herbology, 3: Defense Against the Dark Arts, 4: Divination, 5: Muggle Studies, 6: Ancient Runes, 7: History of Magic, \
			8: Transfiguration, 9: Potions, 10: Care of Magical Creatures, 11: Charms, 12: Flying',\
			nargs='+', default = list(range(0,13)), type=int)
	args = parser.parse_args()
	if not args.datafile.endswith(".csv"):
		print("Error: Datafile must be a .csv file")
		sys.exit()
	return (args)

if __name__ == '__main__':
	args = parse_arguments()
	graph = ScatterPairPlot(args)
	graph.pair_plot()