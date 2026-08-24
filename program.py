## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


## Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)