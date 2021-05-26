from classes.logisticregression import LogisticRegression, ModelEvaluation
import argparse
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='.csv file containing the data to test the model')
    parser.add_argument('weights', help='.pkl file containing the weights to predict the results')
    parser.add_argument('-t', '--test', help='Find out your house, requires as argument name of a csv file of student notes for reference', type=str)
    args = parser.parse_args()
    return (args)

def get_weights(file, model):
    with open (file, "rb") as f:
        info = pickle.load(f)
    model.thetas = info["thetas"]
    model.scaler = info["scaler"]
    model.houses = info["houses"]
    
    
if __name__ == "__main__":
    args = parse_arguments()
    args.verbose = False
    model = LogisticRegression(args, train = False)
    get_weights(args.weights, model)
    model.feature_scale_normalise()
    model.add_bias_units()
    model.hypothesis(model.X)
    model.save_predictions("houses.csv")