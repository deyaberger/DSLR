from classes.logisticregression import LogisticRegression, ModelEvaluation
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='.csv file containing the data to train the model')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0)
    parser.add_argument('-lr', '--learning_rate', help='[default = 0.01]', type=float, default=0.01)
    parser.add_argument('-st', '--steps', help='[default = 100]', type=int, default=100)
    parser.add_argument('-tr', '--train_size', help='percentage of the dataset to generate the train dataset [default = 0.7]', type=float, default=0.7)
    parser.add_argument('-f', '--choose_features', help='train logistic model with chosen features', action='store_true')
    parser.add_argument('-p', '--plot', help='display F1 score', action='store_true')
    args = parser.parse_args()
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
    model.save_weights()
    if args.plot == True:
        score.show_graph("F1_score", score.iterations, score.F1_score_total)
    ### TODO: Parsing arguments: option de choose la/les features
    ### TODO: Indicateur features utiles + Calcul et plot de la fonction de cout
    ### TODO: gestion d'erreur
    ### TODO: Fonction de prediction + ameliorer describe

 