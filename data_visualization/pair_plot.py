import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sys
import argparse

START = 6
END = 19

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to test the model', type=str)
	parser.add_argument('-c', '--courses', help='pair plot from index to index', default="[6,19]", type=str)
	args = parser.parse_args()
	if not args.datafile.endswith(".csv"):
		print("Error: Datafile must be a .csv file")
		sys.exit()
	return (args)

def get_data(datafile, start=START, end=END):
    f = open(datafile, "r")
    csv_reader = csv.reader(f, delimiter=',')
    header = next(csv_reader)
    dataset = []
    for row in csv_reader:
        student = {"house": row[1]}
        for i in range (start, end):
            if row[i]:
                student[header[i]] = float(row[i])
        dataset.append(student)
    return (dataset)

def scatter_plot(datafile, x_value, y_value):
    dataset = get_data(datafile)
    df = pd.DataFrame(dataset)
    sns.scatterplot(data=df, x=x_value, y=y_value, hue="house")
    plt.show()

def main():
    args = parse_arguments()
    if (len(sys.argv) > 2 and int(sys.argv[1]) > 0 and int(sys.argv[1]) < 13):
        dataset = get_data(args.datafile, int(sys.argv[1]) + 6, (int)(sys.argv[2]) + 7)
    else:
        dataset = get_data(args.datafile)
    df = pd.DataFrame(dataset)
    sns.pairplot(df, hue='house')
    plt.title("Pair plot de l'ensemble des cours de Poudlard")
    plt.show()

if __name__ == '__main__':
    ### TODO: titres + question sur plots, label des axes, datafile en argument, parsing arguments, gestion d'erreur arguments
    main()
