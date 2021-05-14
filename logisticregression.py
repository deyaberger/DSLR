import pandas as pd
import numpy as np

class LogisticRegression:
    
    def __init__(self):
        self.X = []
        self.y = []
        self.features = []
        self.thetas = []
        
    def read_csv(self, datafile):
        try:
            df = pd.read_csv(datafile)
            features = list(df.columns[6:])
            self.X = df[features].to_numpy()
            self.y = df["Hogwarts House"].to_numpy()
        except: #FileNotFoundError as e:
            # print(e)
            raise FileNotFoundError('[Read error] Wrong file format. Make sure you give an existing .csv file as argument.')

try:
    test = LogisticRegression()
    X, y, features = test.read_csv("datasets/hey.csv")
except TypeError as e:
    print(e)
    