#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:26:11 2020

@author: derek
"""


import numpy as np
import assignment2_matrix_functions as amf
import assignment2_linear_regression_functions as lirf
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score



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
    

def exercise_1():
    print ("\nExercise 7.1")