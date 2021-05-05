import matplotlib.pyplot
import csv
import matplotlib.pyplot as plt
import sys

DATA_FILE = "../datasets/dataset_train.csv"
COURSE = "Arithmancy"

def house_value(house):
    if house == "Gryffindor":
        return(0)
    elif house == "Hufflepuff":
        return (1)
    elif house == "Ravenclaw":
        return (2)
    else:
        return (3)

def make_data_table(datafile, course):
    f = open(datafile, "r")
    dataset = ([], [], [], [])
    csv_reader = csv.reader(f, delimiter=',')
    header = next(csv_reader)
    index = 6
    for i in range(6,19):
        if header[i] == course:
            index = i
    for row in csv_reader:
        if row[index]:
            note = float(row[index])
            dataset[house_value(row[1])].append(note)
    return(dataset)

def show_histo(datafile, course):
    dataset = make_data_table(datafile, course)
    n_bins = 10
    colors = ['#A8221F', '#CEA622', '#078DBB', '#288747']
    labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    plt.hist(dataset, n_bins, histtype='bar', color=colors, label=labels)
    plt.legend(frameon=False, prop={'size':10})
    plt.title(course)

    plt.show()

def main():
    course = COURSE
    if (len(sys.argv) > 1):
       course = sys.argv[1]
    show_histo(DATA_FILE, course)

if __name__ == '__main__':
    main()
