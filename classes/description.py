import pandas as pd
import numpy as np

class Feature:
    def __init__(self, X, list_params):
        '''
        X: Vector of input
        y: Vector for future output
        '''
        self.X = sorted(X)
        self.y = np.zeros((len(list_params), 1))
        self.count = len(X)
        if self.count != 0:
            self.mean = sum(X) / self.count
            self.std = self.calc_std()
        self.calculate_parameters(list_params)
        

    def calculate_parameters(self, list_params):
        for i, parameter in enumerate(list_params):
            if parameter == "count":
                self.y[i] = self.count
            elif parameter != "count" and self.count == 0:
                self.y[i] = np.NaN
            elif parameter == "mean":
                self.y[i] = self.mean
            elif parameter == "std":
                self.y[i] = self.std
            elif parameter == "min":
                self.y[i] = self.X[0]
            elif parameter == "max":
                self.y[i] = self.X[-1]
            elif parameter == "skw":
                self.y[i] = self.calc_skw()
            else:
                try:
                    quartile = int(parameter[:parameter.find("%")])
                    self.y[i] = self.calc_percentiles(quartile)
                except:
                    print(parameter)
                    print("issue while converting quartile")
    

    def calc_std(self):
        '''
        Standard Deviation (ecart type): Calculates the amount of dispersion of a set of values.
        Answers to the question : how much the values tend to be close to the mean?
        '''
        std = ((sum((self.X - self.mean) ** 2)) / (self.count - 1)) ** 0.5 if self.count > 1 else np.NaN
        return std
    
    def calc_skw(self):
        '''
        Skewness: Measures the asymmetry of the probability distribution of a real-valued random variable about its mean.
        For example, a zero value means that the tails on both sides of the mean balance out overall.
        '''
        skw = sum(((self.X - self.mean) / self.std) ** 3) / (self.count - 1)
        return skw
    
    def calc_percentiles(self, quartile):
        '''
        Percentile: It is a score below which a given percentage of scores in its frequency distribution falls.
        '''
        position_floaty = (float(quartile) / 100) * (self.count - 1)
        min_position = int(position_floaty)
        max_position = min_position + 1
        max_coef = position_floaty - min_position
        if max_coef == 0.0:
            return self.X[min_position]
        min_coef = 1 - max_coef
        result_min = (self.X[min_position] * min_coef)
        result_max = (self.X[max_position] * max_coef)
        return result_min + result_max 



class Describe:
    def __init__(self, args):
        self.empty = False
        self.list_params = args.list_params
        self.read_csv(args.datafile)
        if self.empty == False:
            self.init_output_df()
            self.fill_output_df()
    
    def read_csv(self, datafile):
        '''
        Reading CSV file, keeping only columns that contain numbers in their rows
        M: Matrix of shape (features, list_parameters)
        '''
        try:
            df = pd.read_csv(datafile)
            self.features = list(df.select_dtypes(exclude=['object']).columns)
            self.M = df[self.features].to_numpy().T
            if len(df) == 0:
                self.handle_empty(df)
        except FileNotFoundError:
            print(f"No such file or directory: '{datafile}'")
        except pd.errors.EmptyDataError:
            print(f"No columns to parse from file: '{datafile}'")
    
    def handle_empty(self, df):
        '''
        If the input dataframe only contains a row for columns, the output of describe is slightly different
        '''
        self.empty = True
        cols = len(df.columns)
        data = np.array([[0] * cols, [0] * cols, [np.NaN] * cols, [np.NaN] * cols])
        self.output_df = pd.DataFrame(data = data, index = ["count", "unique", "top", "freq"], columns = df.columns)

    def init_output_df(self):
        '''
        Initializing an output dataframe with only the names for its rows
        '''
        self.output_df = pd.DataFrame(data = None, index = self.list_params)
        
    def clear_empty_values(self, X):
        '''
        For each feature, droping the rows that contain empty or NaN value
        '''
        X=X[np.logical_not(np.isnan(X))]
        return(X)

    def fill_output_df(self):
        for index, feature_name in enumerate(self.features):
            X = self.clear_empty_values(self.M[index])
            feature = Feature(X, self.list_params)
            self.output_df[feature_name] = feature.y