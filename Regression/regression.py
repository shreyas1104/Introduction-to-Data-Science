# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:29:05 2021

@author: Dell
"""

# Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_excel(r'E:\Learning-Courses\Intro to Data Science\Regression\Folds5x2_pp.xlsx')
x = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fitting Gradient Boosting to the Training set
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
