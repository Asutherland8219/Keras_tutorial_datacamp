import pandas as pd
import matplotlib.pyplot as plt
import numpy as npp
from sklearn.model_selection import train_test_split


#read in the data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

# Pre process the data and begin creating the information for the neural network. This is where we are labeling the data (ie. hot encoding ) and joining the data

# add a type column for red with value 1
red['type'] = 1

# add a type column  for white with a value of 0
white['type'] = 0

# Append 'white' to 'red' 
wine = red.append(white, ignore_index=True)

print(wine.head())

# Use the created data to come up with a machine learning model specfically using sklearn

# Specify the data
X = wine.ix[:,0:11]

# Specify the target labels and flatten the array
y = np.ravel(wine.type)

# Split the data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)