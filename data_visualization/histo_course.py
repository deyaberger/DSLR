import csv
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np

COURSE = "Arithmancy"

class PlotHistogram:

    def __init__(self, args):
        self.datafile = args.datafile
        self.course = args.course

    def house_value(self, house):
        if house == "Gryffindor":
            return(0)
        elif house == "Hufflepuff":
            return (1)
        elif house == "Ravenclaw":
            return (2)
        else:
            return (3)

    def make_data_table(self):
        f = open(self.datafile, "r")
        self.dataset = ([], [], [], [])
        csv_reader = csv.reader(f, delimiter=',')
        header = next(csv_reader)
        exist = False
        index = 6
        for i in range(6,19):
            if header[i] == self.course:
                exist = True
                index = i
        if (exist == False):
            print(f"{self.course} does not exist, ploting Arithmancy instead.")
            self.course = "Arithmancy"
        for row in csv_reader:
            if row[index]:
                note = float(row[index])
                self.dataset[self.house_value(row[1])].append(note)

    def show_histo(self):
        self.make_data_table()
        n_bins = 10
        colors = ['#A8221F', '#CEA622', '#078DBB', '#288747']
        labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        plt.hist(self.dataset, n_bins, histtype='bar', color=colors, label=labels)
        plt.legend(frameon=False, prop={'size':10})
        plt.title(f"Quel cours de Poudlard a une répartition des notes homogènes entre les quatres maisons ? {self.course}")
        plt.show()
