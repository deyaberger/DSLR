from classes import LogisticRegression, TestHouses
import sys
try:
    import argparse
    import pickle
    import numpy as np
except ModuleNotFoundError as e:
    print(e)
    print("Please launch python -r requirements.txt")
    sys.exit()

def display_error(msg):
    print(msg)
    sys.exit()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='.csv file containing the data to test the model', type=str)
    parser.add_argument('weights', help='.pkl file containing the weights to predict the results', type=str)
    parser.add_argument('-f', '--find', help='Find out your house, requires as argument name of a csv file of student notes for reference', type=str)
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
        if type(structure["houses"]) != list or type(structure["features"]) != list\
            or type(structure["activation"]) != str or type(structure["thetas"]) != np.ndarray:
            display_error("Wrong format for weigths file")
    except ValueError as e:
        display_error(e)
    return (structure)

def get_weights(structure, model):
    try:
        model.thetas = structure["thetas"]
        model.houses = structure["houses"]
        model.features = structure["features"]
    except:
        display_error("Wrong format for weigths file")

    
if __name__ == "__main__":
    args = parse_arguments()
    structure = get_model_structure(args.weights)
    args.activation = structure["activation"]
    args.features = structure["features"]
    model = LogisticRegression(args, train = False)
    get_weights(structure, model)
    model.feature_scale_normalise()
    model.add_bias_units()
    if args.verbose == 1:
        print("- Predicting results -\n")
    model.hypothesis(model.X)
    model.save_predictions("houses.csv")
    if args.find:
        args.houses = model.houses
        args.thetas = model.thetas
        find = TestHouses(args)
        find.launch_test()