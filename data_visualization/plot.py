import sys
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
	print(e)
	print("Please launch 'python -r requirements.txt'")
	sys.exit()

class ScatterPairPlot:

	def __init__(self, args):
		self.datafile = args.datafile
		self.args = args

	def get_data(self):
		try:
			self.dataset = pd.read_csv(self.datafile)
		except:
			print("Can't parse csv file.")
			sys.exit()
		if not "Hogwarts House" in self.dataset:
			print("Can't find student's houses")
			sys.exit()

	def scatter_plot(self):
		self.get_data()
		if not self.args.x_course in self.dataset:
			print(f"{self.args.x_course} not found")
			sys.exit()
		if not self.args.y_course in self.dataset:
			print(f"{self.args.y_course} not found")
			sys.exit()
		sns.scatterplot(data=self.dataset, x=self.args.x_course, y=self.args.y_course, hue="Hogwarts House")
		plt.title(f"Quelles sont les deux features qui sont semblables ?\n{self.args.x_course}, {self.args.y_course}")
		plt.show()

	def pair_plot(self):
		self.get_data()
		if self.dataset.empty:
			print("File is empty")
			sys.exit()
		if self.dataset['Hogwarts House'].isnull().values.all():
			print("Students have no house. You probably used the testing set instead of the training one")
			sys.exit()
		try :
			lst_var = [self.dataset.columns[i + 6] for i in self.args.features]
		except:
			print("Incorrect parameters.")
			sys.exit()
		try:
			sns.pairplot(self.dataset, vars=lst_var, hue='Hogwarts House')
		except:
			print("Can't plot pairplot, this is likely due because x variable is categorical but one of ['numeric', 'datetime'] is required")
			sys.exit()
		plt.show()