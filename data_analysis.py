import pandas as pd
import math
import numpy as np
from colorama import Fore
from colorama import Style
from IPython.display import display

class Describe:
    def __init__(self, dataset):
        self.sorted_dataset = sorted(dataset)
        self.count = len(dataset)
        self.mean = sum(dataset) / self.count
        self.std = self.calc_std(dataset)
        self.min = self.sorted_dataset[0]
        self.p_25 = self.calc_percentiles(25)
        self.p_50 = self.calc_percentiles(50)
        self.p_75 = self.calc_percentiles(75)
        self.max = self.sorted_dataset[-1]
    
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
    
class PandasFeature:
    def __init__(self, dataset):
        p_25_index = 4
        p_50_index = 5
        p_75_index = 6
        self.count = dataset.count()
        self.mean = dataset.mean()
        self.std = dataset.std()
        self.min = dataset.min()
        self.p_25 = dataset.describe()[p_25_index]
        self.p_50 = dataset.describe()[p_50_index]
        self.p_75 = dataset.describe()[p_75_index]
        self.max = dataset.max()
        
def fill_final(final_df):
    final_df.at["count", "Mine"], final_df.at["count", "Pandas"], final_df.at["count", "Compare"] = elmio.count, df_feature.count, (round(elmio.count, 4) == round(df_feature.count, 4))
    final_df.at["mean", "Mine"], final_df.at["mean", "Pandas"], final_df.at["mean", "Compare"] = elmio.mean, df_feature.mean, (round(elmio.mean, 4) == round(df_feature.mean, 4))
    final_df.at["std", "Mine"], final_df.at["std", "Pandas"], final_df.at["std", "Compare"] = elmio.std, df_feature.std, (round(elmio.std, 4) == round(df_feature.std, 4))
    final_df.at["min", "Mine"], final_df.at["min", "Pandas"], final_df.at["min", "Compare"] = elmio.min, df_feature.min, (round(elmio.min, 4) == round(df_feature.min, 4))
    final_df.at["25%", "Mine"], final_df.at["25%", "Pandas"], final_df.at["25%", "Compare"] = elmio.p_25, df_feature.p_25, (round(elmio.p_25, 4) == round(df_feature.p_25, 4))
    final_df.at["50%", "Mine"], final_df.at["50%", "Pandas"], final_df.at["50%", "Compare"] = elmio.p_50, df_feature.p_50, (round(elmio.p_50, 4) == round(df_feature.p_50, 4))
    final_df.at["75%", "Mine"], final_df.at["75%", "Pandas"], final_df.at["75%", "Compare"] = elmio.p_75, df_feature.p_75, (round(elmio.p_75, 4) == round(df_feature.p_75, 4))
    final_df.at["max", "Mine"], final_df.at["max", "Pandas"], final_df.at["max", "Compare"] = elmio.max, df_feature.max, (round(elmio.max, 4) == round(df_feature.max, 4))

def good_or_not(val):
    if (val == True and val.dtype == bool):
        color = 'green'
    elif (val == False and val.dtype == bool):
        color = 'red'
    else:
        color = 'grey'
    return 'color: %s' % color

if __name__ == "__main__":
    df = pd.read_csv("datasets/dataset_test.csv")
    feature_name = 'Arithmancy'
    dataset = df[feature_name].to_numpy()
    dataset = dataset[~np.isnan(dataset)]
    elmio = Describe(dataset)
    df_feature = PandasFeature(df[feature_name])
    final_df = pd.DataFrame(index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"], columns=["Mine", "Pandas", "Compare"])
    fill_final(final_df)
    colored_final_df = final_df.style.applymap(good_or_not)
    final_df.style