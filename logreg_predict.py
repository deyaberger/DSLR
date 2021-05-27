from classes.logisticregression import LogisticRegression, ModelEvaluation
from classes.test import TestHouses
import argparse
import pickle
import sys

def display_error(msg):
	print(msg)
	sys.exit()

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to test the model', type=str)
	parser.add_argument('weights', help='.pkl file containing the weights to predict the results', type=str)
	parser.add_argument('-t', '--test', help='Find out your house, requires as argument name of a csv file of student notes for reference', type=str)
	parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
	args = parser.parse_args()
	if not args.datafile.endswith(".csv"):
		display_error("Error: Datafile must be a .csv file")
	if not args.weights.endswith(".pkl"):
		display_error("Error: Weights file must be a .pkl file")
	return (args)

def get_model_structure(file):
	try:
		with open (file, "rb") as f:
			structure = pickle.load(f)
	except ValueError as e:
		display_error(e)
	return (structure)

def get_weights(structure, model):
	model.thetas = structure["thetas"]
	model.houses = structure["houses"]
	model.features = structure["features"]

	
if __name__ == "__main__":
	args = parse_arguments()
	structure = get_model_structure(args.weights)
	args.activation = structure["activation"]
	model = LogisticRegression(args, train = False)
	get_weights(structure, model)
	model.feature_scale_normalise()
	model.add_bias_units()
	model.hypothesis(model.X)
	model.save_predictions("houses.csv")
	if args.test:
		args.houses = model.houses
		args.thetas = model.thetas
		test = TestHouses(args)
		test.launch_test()

## TODO : titres plos, steps = epochs, performance en fonction epochs