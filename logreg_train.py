from classes.logisticregression import LogisticRegression, ModelEvaluation
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('datafile', help='.csv file containing the data to train the model')
	parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
	parser.add_argument('-a', '--activation', help='Choose activation function', type=str, default="sigmoid", choices = ["sigmoid", "softmax"])
	parser.add_argument('-lr', '--learning_rate', help='[default = 0.01]', type=float, default=0.01)
	parser.add_argument('-ep', '--epochs', help='Number of iterations, must be between 1 and 200, [default = 100]', type=int, default=100)
	parser.add_argument('-tr', '--train_size', help='percentage of the dataset to generate the train dataset [default = 0.7]', type=float, default=0.7)
	parser.add_argument('-f', '--features', help='train logistic model with chosen features : 0 = Arithmancy\, 1 = Astronomy, 2 = Herbology, 3_Defense against the Dark Arts, 4_Divination, 5_Muggle Studies, 6_Ancient Runes, 7_History of Magic, 8_Transfiguration, 9_Potions, 10_Care of Magical Creatures, 11_Charms, 12_Flying', nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12], type=int)
	parser.add_argument('-c', '--coefficients', help='display coefficents value', action='store_true')
	parser.add_argument('-p', '--plot', help='display F1 score', action='store_true')
	parser.add_argument('-sch', '--stochastic', help='Compute stochastic gradient descent', action='store_true')
	args = parser.parse_args()
	if args.epochs > 200:
		args.epochs = 200
	if args.epochs <= 0:
		args.epochs = 1
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
		score.show_graph("F1_score", score.iterations, score.F1_score_total)
	if args.coefficients == True:
		model.display_coefficients('weights.pkl')
	### TODO: Parsing arguments: option de choose la/les features
	### TODO: Indicateur features utiles + Calcul et plot de la fonction de cout
	### TODO: gestion d'erreur
 