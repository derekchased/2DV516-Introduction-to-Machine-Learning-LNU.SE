#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:06:31 2020

@author: derek
"""

import numpy as np
import assignment2_matrix_functions as amf
import assignment2_linear_regression_functions as lirf

"""
import matplotlib.pyplot as plt
import assignment2_logistic_regression_functions as lorf
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
"""
def load_data():
    # Step 1 - Load Data
    Csv_data = np.loadtxt("./A2_datasets_2020/GPUBenchmark.csv",delimiter=',') # load csv
    
    # Step 2 - Setup X and y
    X = Csv_data[:,:-1]
    y = Csv_data[:,6]
    
    # Step 3 - Normalize Data
    Xn = amf.feature_normalization(X)
    
    # Step 4 - Return normalized X and labels
    return Xn, y
    

def exercise6_1():
    print ("\nExercise 6.1")

def cost(X,y,j):
    Xreduced = X[:,j-1].reshape(-1,1)
    #print(Xreduced)
    #print(Xreduced.shape)
    #2 - Extend Xn
    Xe = amf.extended_matrix(Xreduced)
        
    # 3 - Get betas using normal equation
    betas = lirf.normal_equation(Xe,y)
    
    # 4 - Get Cost
    normal_eq_cost = lirf.cost_function(Xe,betas,y)
    #print("COST:: j\t",j,"j_cost\t",normal_eq_cost)
    return normal_eq_cost

    
#   Forward selection using breadth first recursion
def compare(X, y, j=0):
    #   End when you've got to the last column. Return the index and the cost
    if j >= Xn.shape[1]:
        #print("if",j)
        return j, cost(X,y,j)
    else:
        #   Recurse over j+1
        jplusone, jplusone_cost = compare(X,y,j+1) 
        
        #   Calculate the current cost
        j_cost = cost(X,y,j)
        
        #print("j1\t",jplusone," j+1C\t",jplusone_cost)
        #print("j\t",j,"jC\t",j_cost)
        
        #   Compare j cost to the best j+1 cost, return whichever is best
        if j_cost <= jplusone_cost:
            return j, j_cost
        else:
            return jplusone, jplusone_cost

# 1 - Load Data    
Xn, y = load_data()
j, cost = compare(Xn,y)





"""
# Load the diabetes dataset
#X, y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
#X = X[:, np.newaxis, 2]

# Split the data into training/testing sets
#X_train = X[:-20]
#X_test = X[-20:]
X_train = X
X_test = X

# Split the targets into training/testing sets
#y_train = y[:-20]
#y_test = y[-20:]
y_train = y
y_test = y

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, y_pred, color='blue', linewidth=3)
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, y_pred, color='blue', linewidth=3)


plt.xticks(())
plt.yticks(())

plt.show()

ddd = X*np.array([ [0,1 ], [0,1 ]   ])
ddd2 = X*np.array([ [1,0 ], [1,0 ]   ])
ddd3 = X*np.array([ [1,1 ], [1,1 ]   ])
"""


    

