import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import errors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import pickle

def display_error(msg):
    print(msg)
    sys.exit()

class ModelEvaluation:
    def __init__(self):
        ''' Validation metrics:
        True positive: correct prediction for a student to be part of a certain house in Hogwart
        True negative: correct prediction for a student to NOT be part of a certain house in Hogwart
        False positive: incorrect prediction for a student to be part of a certain house in Hogwart
        False negative: incorrect prediction for a student to NOT be part of a certain house in Hogwart
        '''
        self.true_positive = np.zeros((4, 1))
        self.true_negative = np.zeros((4, 1))
        self.false_negative = np.zeros((4, 1))
        self.false_positive = np.zeros((4, 1))
        self.precision_total, self.sensitivity_total, self.specificity_total, self.accuracy_total, self.F1_score_total = [], [], [], [], []
        self.iterations = []
    
    def calculate_kpi(self):
        ''' Commonly use KPIs:
        Precision: expresses the proportion of the data points our model says was relevant actually were relevant.
        Sensitivity: (also called redcall) expresses the ability to find all relevant instances in a dataset.
        Specificity: refers to how well our model identifies our students to not be part of a certain house.
        Accuracy: proportion of correct predictions over total predictions.
        F1 Score: the harmonic mean of the model s precision and recall.
        '''
        self.precision = self.my_divide(self.true_positive, (self.true_positive + self.false_positive))
        self.sensitivity = self.my_divide(self.true_positive, (self.true_positive + self.false_negative))
        self.specificity = self.my_divide(self.true_negative, (self.true_negative + self.false_positive))
        self.accuracy = self.my_divide((self.true_positive + self.true_negative), (self.true_positive + self.true_negative + self.false_positive + self.false_negative))
        self.F1_score = (2 * (self.my_divide((self.precision * self.sensitivity), (self.precision + self.sensitivity))))
    
    def save_evolution(self, steps):
        ''' Saving all values of our kpis during each call to the evaluate function, to display them in the graphs '''
        self.precision_total.append(np.mean(self.precision) * 100)
        self.sensitivity_total.append(np.mean(self.sensitivity) * 100)
        self.specificity_total.append(np.mean(self.specificity) * 100)
        self.accuracy_total.append(np.mean(self.accuracy) * 100)
        self.F1_score_total.append(np.mean(self.F1_score) * 100)
        self.iterations.append(steps)

    def evaluate(self, model, steps, X, y):
        '''
        Calculating different type of KPIs to evaluate the performance of our model and its evolution
        '''
        model.hypothesis(X)
        for i in range(len(X)):
            predicted_house, real_house = model.predict(y, i)
            if predicted_house == real_house:
                self.true_positive[predicted_house] += 1
                for index, nb in enumerate(model.H[i]):
                    if index != i:
                        self.true_negative[index] += 1
            else:
                self.false_positive[predicted_house] += 1
                self.false_negative[real_house] += 1
        self.calculate_kpi()
        self.save_evolution(steps)

    def my_divide(self, a, b):
        result = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        return(result)

    def show_graph(self, name, x, y):
        plt.title(f"{name} evolution")
        plt.xlabel("iterations")
        plt.ylabel(name)
        plt.plot(x, y)
        plt.show()

class LogisticRegression:
    
    def __init__(self, args = None, train = True):
        self.args = args
        self.read_csv(self.args.datafile, train)
        self.activation = self.sigmoid
    
    def check_input(self, df, features, train):
        if features == []:
            display_error("Missing features for our training")
        if train == True and (self.y.any() == False or sum(pd.isnull(df["Hogwarts House"]))):
            display_error("Missing data in the Hogwarts House column")
        if self.y.shape[0] != self.X.shape[0]:
            display_error("Error in dataset, matrixes cannot be used for our logistic regression")
        try:
            np.isnan(self.X.astype(np.float))
        except ValueError as e:
            display_error(e)
   

    def read_csv(self, datafile, train = True):
        ''' Reading our Training data
        Features: only numeric values (not taking in account the names of the students for example)
        X will be our matrix of students and their grades the different subjects
        y will be the houses of these students
        For each missing values in grades, we replace it by the median grade of all students
        '''
        try:
            df = pd.read_csv(datafile)
            df.fillna(df.median(), inplace = True)
            self.features = list(df.columns[6:]) ### TODO change features
            self.X = df[self.features].to_numpy()
            one_hot_encoding = pd.get_dummies(df["Hogwarts House"], drop_first = False)
            self.houses = list(one_hot_encoding.columns)
            self.y = one_hot_encoding.to_numpy()
            self.check_input(df, self.features, train)
        except (FileNotFoundError, errors.EmptyDataError) as e:
            display_error(e)
        except KeyError as e:
            display_error(f"The csv file does not contain the expected column {e}")
    
    def feature_scale_normalise(self):
        '''
        Scaling our data so the mean of each feature will be 0 - helps to converge faster
        '''
        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

    def add_bias_units(self):
        '''
        Adding as the first column of our matrice, a vector of size = number of students, values = 1
        So that we can have theta0 as our bias:
        theta0 * 1 + theta1 * feature1 + theta2 * feature2 etc...
        '''
        bias_units = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((bias_units, self.X), axis = 1)
    
    def split_data(self):
        ''' Splitting our data in a training set and a testing set, so when we calculate our score, we avoid bias and overfitting
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.args.train_size, random_state=42)
    
    def init_weights(self):
        ''' Initializing our thetas to 0
        '''
        self.thetas = np.zeros((self.X_train.shape[1], self.y_train.shape[1]))
    

    def sigmoid(self, x):
        ret = 1 / (1 + np.exp(-x))
        return(ret)

    def hypothesis(self, X):
        '''
        Our prediction must be 1 or 0 (being part of a certain house or not)
        z: predicts a number
        --> The activation function transforms this number into a "probability" between 0 and 1
        '''
        z = np.matmul(X, self.thetas)
        self.H = self.activation(z)
    
    def compute_loss_gradient(self, X, y):
        '''
        H[0] = list of probalities of student number 0 to be part of the different Hogwart House
        H - y : difference between the prediction and the "truth"
        loss_gradient: derivative of our cost function
        '''
        error = self.H - y
        self.loss_gradient = np.matmul(X.T, error) / len(X)

    def gradient_descent(self):
        '''
        Changes the values of the thetas according to the derivative of our cost function
        '''
        self.thetas = self.thetas - (self.args.learning_rate * self.loss_gradient)
    
    def predict(self, y, i):
        '''
        For a student of index i in the dataset, returns the house (an index of model.houses) of the prediction and of the truth
        '''
        predicted_house = np.argmax(self.H[i])
        real_house = np.argmax(y[i])
        return (predicted_house, real_house)
    
    def fit(self, score):
        '''
        Calculate our predictions, then compute loss gradient according to our error
        '''
        for i in range(self.args.steps):
            score.evaluate(self, i, self.X_test, self.y_test)
            self.hypothesis(self.X_train)
            self.compute_loss_gradient(self.X_train, self.y_train)
            self.gradient_descent()
    
    def save_weights(self, file_name):
        info = {"thetas" : self.thetas, "scaler" : self.scaler, "houses" : self.houses}
        with open(file_name, "wb") as f:
            pickle.dump(info, f)

    def create_df_predictions(self, predictions, predictions_file):
        df = pd.DataFrame(data = predictions, columns = ["Hogwarts House"])
        df.index.name = "Index"
        df.to_csv(predictions_file, index = True)
 
    def save_predictions(self, predictions_file):
        predictions = []
        for i in range(len(self.H)):
            house = self.houses[np.argmax(self.H[i])]
            predictions.append(house)
        self.create_df_predictions(predictions, predictions_file)