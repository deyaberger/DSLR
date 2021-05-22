import pandas as pd
import numpy as np
import argparse

class Feature:
    def __init__(self, X, list_params):
        self.X = sorted(X)
        self.count = len(X)
        if self.count != 0:
            self.mean = sum(X) / self.count
        self.calculate_parameters(list_params)

    def calculate_parameters(self, list_params):
        self.y = np.zeros((len(list_params), 1))
        for i, parameter in enumerate(list_params):
            if parameter == "count":
                self.y[i] += self.count
            elif parameter == "mean":
                self.y[i] += self.mean
            elif parameter == "std":
                self.y[i] += self.calc_std()
            elif parameter == "min":
                self.y[i] += self.X[0]
            elif parameter == "max":
                self.y[i] += self.X[-1]
            else:
                try:
                    quartile = int(parameter[:parameter.find("%")])
                    self.y[i] += self.calc_percentiles(quartile)
                except:
                    print("issue while converting quartile")
    
    def calc_std(self):
        std = ((sum(np.square(self.X - self.mean))) / (self.count - 1)) ** 0.5
        return std
    
    def calc_percentiles(self, quartile):
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
        self.list_params = args.list_params
        self.read_csv(args.datafile)
        self.init_output_df()
        self.fill_output_df()
    
    def read_csv(self, datafile):
        try:
            df = pd.read_csv(datafile)
            self.features = list(df.select_dtypes(exclude=['object']).columns)
            self.M = df[self.features].to_numpy().T
        except FileNotFoundError:
            print(f"No such file or directory: '{datafile}'")
        except pd.errors.EmptyDataError:
            print(f"No columns to parse from file: '{datafile}'")
        return (None)

    def init_output_df(self):
        self.output_df = pd.DataFrame(data = None, index = self.list_params)
        
    def clear_empty_values(self, X):
        X=X[np.logical_not(np.isnan(X))]
        return(X)

    def fill_output_df(self):
        for index, feature_name in enumerate(self.features):
            X = self.clear_empty_values(self.M[index])
            feature = Feature(X, self.list_params)
            self.output_df[feature_name] = feature.y
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='.csv file containing the data to describe')
    parser.add_argument('-f', '--full_display', help='display all rows of the describe dataframe', action='store_true')
    parser.add_argument('-s', '--save', help='save the info of describe in a csv file', action='store_true')
    parser.add_argument('-q', '--quartile', help='Calculate additional quartiles', action = 'append', type = int, choices = list(range(0, 101)))
    args = parser.parse_args()
    args.list_params = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    if args.quartile:
        for q in args.quartile:
            name = str(q) + "%"
            args.list_params.append(name)
    return (args)

if __name__ == "__main__":
    args = parse_arguments()
    print(args.quartile)
    describe = Describe(args)
    if args.full_display == True:
        print(describe.output_df.to_string())
    else:
        print(describe.output_df)
    if args.save == True:
        describe.output_df.to_csv("describe.csv", index=False)
    