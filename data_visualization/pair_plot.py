import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sys

DATAFILE = "../datasets/dataset_train.csv"
START = 6
END = 19

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
    dataset = get_data(DATAFILE)
    df = pd.DataFrame(dataset)
    sns.scatterplot(data=df, x=x_value, y=y_value, hue="house")
    plt.show()

def main():
    if (len(sys.argv) > 2 and int(sys.argv[1]) > 0 and int(sys.argv[1]) < 13):
        dataset = get_data(DATAFILE, int(sys.argv[1]) + 6, (int)(sys.argv[2]) + 7)
    else:
        dataset = get_data(DATAFILE)
    df = pd.DataFrame(dataset)
    sns.pairplot(df, hue='house')
    plt.show()

if __name__ == '__main__':
    ### TODO: titres + question sur plots, label des axes, datafile en argument, parsing arguments, gestion d'erreur arguments
    main()
