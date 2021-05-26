from classes.logisticregression import LogisticRegression, ModelEvaluation
from classes.test import TestHouses
import argparse
import pickle

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to test the model')
	parser.add_argument('weights', help='.pkl file containing the weights to predict the results')
	parser.add_argument('-t', '--test', help='Find out your house, requires as argument name of a csv file of student notes for reference', type=str)
	parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
	args = parser.parse_args()
	return (args)

def get_weights(file, model):
	with open (file, "rb") as f:
		info = pickle.load(f)
	model.thetas = info["thetas"]
	model.scaler = info["scaler"]
	model.houses = info["houses"]

def get_features(file):
	with open (file, "rb") as f:
		info = pickle.load(f)
	return(info["features"])
	
if __name__ == "__main__":
	args = parse_arguments()
	args.features = get_features(args.weights)
	model = LogisticRegression(args, train = False)
	get_weights(args.weights, model)
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