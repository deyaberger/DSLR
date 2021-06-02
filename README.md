# DSLR

## I. Data Analysis
Run ```python3 describe.py dataset.csv``` to get the description of a dataset. </br>Use ```-h``` to display the usage and the options

## II. Data Visualization

1. Run ```python3 data_visualization/histogram.py``` to display the histogram that answers the question:</br>*Which Hogwarts class has an homogenous repartition of grades between the four houses ?* (-h for more options)

2. Run ```python3 data_visualization/scatter_plot.py``` to display a scatter plot that answers the following question:</br>*Which are the 2 similar features ?* (-h for more options)

3. Run ```python3 data_visualization/pair_plot.py``` to display a pair plot that answers the following question:</br>*Which are the features we are going to use in our training ?* (-h for more options)

## III.Logistic Regression

1. Run ```python3 python3 logreg_train.py datasets/dataset_train.csv``` to train the model. It should creates a file called "weights.pkl" that will be used in the prediction program.</br>Use ```-h``` to display the usage and the options

2. Run ```python3 logreg_predict.py datasets/dataset_test.csv``` to predict the houses for students of the test dataset. It should create a csv file called "house.csv" where all the predictions are saved.</br>Use ```-h``` to display the usage and the options

