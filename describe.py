import pandas as pd
import numpy as np
from IPython.display import display
import random

class Feature:
    def __init__(self, name, dataset):
        self.name = name
        self.sorted_dataset = sorted(dataset)
        self.count = len(dataset)
        if self.count != 0:
            self.mean = sum(dataset) / self.count
            self.std = self.calc_std(dataset)
            self.min = self.sorted_dataset[0]
            self.p_25 = self.calc_percentiles(25)
            self.p_50 = self.calc_percentiles(50)
            self.p_75 = self.calc_percentiles(75)
            self.max = self.sorted_dataset[-1]
            self.infos = [self.count, self.mean, self.std, self.min, self.p_25, self.p_50, self.p_75, self.max]
        else:
            self.infos = [self.count]
            self.infos.extend([np.nan] * 7)
    
    def calc_std(self, dataset):
        sum_squares = 0
        for i in range(len(dataset)):
            sum_squares += (dataset[i] - self.mean) ** 2
        std = sum_squares / (self.count - 1)
        std = std ** 0.5
        return std
    
    def calc_percentiles(self, quartile):
        position_floaty = (float(quartile) / 100) * (self.count - 1)
        min_position = int(position_floaty)
        max_position = min_position + 1
        max_coef = position_floaty - min_position
        if max_coef == 0.0:
            return self.sorted_dataset[min_position]
        min_coef = 1 - max_coef
        result_min = (self.sorted_dataset[min_position] * min_coef)
        result_max = (self.sorted_dataset[max_position] * max_coef)
        return result_min + result_max 

def init_describe_df(list_params):
    output_df = pd.DataFrame(data = None, index = list_params)
    return (output_df)

def parse_arguments(args):
    ### TODO : parse arguments
    dataset_name = "datasets/dataset_test.csv"
    list_params = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    return(dataset_name, list_params)

def clear_empty_values(feature):
    dataset = feature.to_numpy()
    dataset = dataset[~np.isnan(dataset)]
    return (dataset)
    

def fill_output_df(input_df, output_df, list_params):
    for feature_name in input_df:
        if input_df[feature_name].dtype == np.int or input_df[feature_name].dtype == np.float:
            dataset = clear_empty_values(input_df[feature_name])
            feature = Feature(feature_name, dataset)
            output_df[feature_name] = feature.infos

if __name__ == "__main__":
    dataset_name, list_params = parse_arguments(None)
    input_df = pd.read_csv(dataset_name)
    output_df = init_describe_df(list_params)
    fill_output_df(input_df, output_df, list_params)
    display(output_df)