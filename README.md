# DSLR

## Prerequisit:
apt-get install python3</br>
pip install -r requirements.txt
cd src

## I. Data Analysis
Run ```python3 describe.py [a_dataset.csv]``` to get the description of a dataset. </br>Use ```-h``` to display the usage and the options

## II. Data Visualization
Use ```-h``` to display the usage and the options for the following functions:

1. Run ```python3 histogram.py ../datasets/dataset_train.csv``` to display the histogram that answers the question:</br>*Which Hogwarts class has an homogenous repartition of grades between the four houses ?*

2. Run ```python3 scatter_plot.py ../datasets/dataset_train.csv``` to display a scatter plot that answers the following question:</br>*Which are the 2 similar features ?*

3. Run ```python3 pair_plot.py ../datasets/dataset_train.csv``` to display a pair plot that answers the following question:</br>*Which are the features we are going to use in our training ?*

## III.Logistic Regression
Use ```-h``` to display the usage and the options for the following functions:

1. Run ```python3 logreg_train.py ../datasets/dataset_train.csv``` to train the model. It should creates a file called "weights.pkl" that will be used in the prediction program.</br>

2. Run ```python3 logreg_predict.py ../datasets/dataset_test.csv weights.pkl``` to predict the houses for students of the test dataset. It should create a csv file called "house.csv" where all the predictions are saved.</br>

