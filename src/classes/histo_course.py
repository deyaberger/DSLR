import csv
import sys
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
	print(e)
	print("Please launch python -r requirements.txt")
	sys.exit()


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
        elif house == "Slytherin":
            return (3)
        else :
            print("Some students don't have a house attributed")
            sys.exit()

    def check_file(self, header):
        house_index = -1
        default_course = -1
        for i, name in enumerate(header[:]):
            if name == "Hogwarts House":
                house_index = i
            if name == "Arithmancy":
                default_course = i
        if house_index == -1:
            print("Hogwarts House of students not found")
            sys.exit()
        if default_course == -1:
            print("Arithmancy notes are not specified")
            sys.exit()
        return (default_course, house_index)

    def make_data_table(self):
        f = open(self.datafile, "r")
        self.dataset = ([], [], [], [])
        csv_reader = csv.reader(f, delimiter=',')
        try :
            header = next(csv_reader)
        except :
            print("Deficient file")
            sys.exit()
        exist = False
        index, index_house = self.check_file(header)
        for i in range(len(header)):
            if header[i] == self.course:
                exist = True
                index = i
        if (exist == False):
            print(f"{self.course} does not exist, ploting Arithmancy instead.")
            self.course = "Arithmancy"
        for row in csv_reader:
            if row[index]:
                note = float(row[index])
                self.dataset[self.house_value(row[index_house])].append(note)
        if self.dataset == ([], [], [], []):
            print("File contains no data")
            sys.exit()

    def show_histo(self):
        self.make_data_table()
        n_bins = 10
        colors = ['#A8221F', '#CEA622', '#078DBB', '#288747']
        labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        plt.hist(self.dataset, n_bins, histtype='bar', color=colors, label=labels)
        plt.legend(frameon=False, prop={'size':10})
        plt.title(f"Quel cours de Poudlard a une répartition des notes homogènes entre les quatres maisons ? {self.course}")
        plt.xlabel(f"{self.course}'s note")
        plt.ylabel("Number of students")
        plt.show()
