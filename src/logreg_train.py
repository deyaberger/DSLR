from classes import LogisticRegression, ModelEvaluation
import sys
try:
    import argparse
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
	print(e)
	print("Please launch python -r requirements.txt")
	sys.exit()

def display_error(msg):
	print(msg)
	sys.exit()

def check_min_max(args):
	if args.learning_rate <= 0 or args.learning_rate > 1:
		display_error(f"Error: learning rate range = ]0, 1]")
	if args.epochs > 500 or args.epochs < 1:
		display_error(f"Error: for precaution measures, epochs range = [1, 500]")
	if args.train_size < 0.1 or args.train_size > 0.9:
		display_error(f"Error: Train size range = [0.1, 0.9s]")

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to train the model')
	parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
	parser.add_argument('-lr', '--learning_rate', help='[default = 0.01], must be a float betweem 0 and 1', type=float, default=0.01)
	parser.add_argument('-a', '--activation', help='Choose activation function', type=str, default="sigmoid", choices = ["sigmoid", "softmax"])
	parser.add_argument('-sch', '--stochastic', help='Compute stochastic gradient descent', action='store_true')
	parser.add_argument('-ep', '--epochs', help='Number of iterations, must be between 1 and 500, [default = 100]', type=int, default=100)
	parser.add_argument('-tr', '--train_size', help='percentage of the dataset to generate the train dataset [default = 0.7], must be between 0.10 and 0.90', type=float, default=0.7)
	parser.add_argument('-f', '--features', help='train logistic model with chosen features : 0 = Arithmancy, 1 = Astronomy, 2 = Herbology, 3 = Defense against the Dark Arts, 4 = Divination, 5 = Muggle Studies, 6 = Ancient Runes, 7 = History of Magic, 8 = Transfiguration, 9 = Potions, 10 = Care of Magical Creatures, 11 = Charms, 12 = Flying', nargs='+', default = [2,3,4,5,6,7,8,9,11,12], type=int)
	parser.add_argument('-p', '--plot', help='display F1 score and cost function', action='store_true')
	args = parser.parse_args()
	check_min_max(args)
	return (args)


if __name__ == "__main__":
	args = parse_arguments()
	model = LogisticRegression(args)
	model.feature_scale_normalise()
	model.add_bias_units()
	model.split_data()
	model.init_weights()
	score = ModelEvaluation()
	model.fit(score)
	model.save_weights("weights.pkl")
	if args.plot == True:
		score.show_graph("F1_score", list(range(len(score.F1_score_total))), score.F1_score_total, 1)
		score.show_graph("Cost Evolution", list(range(len(model.total_cost))), model.total_cost, 2)
		model.display_weights()
		plt.show() 