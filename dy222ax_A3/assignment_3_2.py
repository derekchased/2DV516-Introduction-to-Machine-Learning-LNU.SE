#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:54:26 2020

@author: derek
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import assignment2_matrix_functions as amf


def load_data():
    print("load_data")
    # Load data from https://www.openml.org/d/554
    
    """
    try:
        X = np.loadtxt('./data/der_mnist_x.csv',delimiter=',')
        y = np.loadtxt("./data/der_mnist_y.csv",delimiter=',')
        
        X = np.load("./data/mnist_x.npy",allow_pickle=True,fix_imports=True)
        y = np.load("./data/mnist_y.npy",allow_pickle=True,fix_imports=True)

    except IOError as error:
        print("IOError",error)
        X,y = fetch()
    except ValueError as error:
        print("ValueError", error)
        X,y = fetch()
    else:
        print("error occurred")
        # Returns data and labels as numpy arrays
        X,y = fetch()"""

    # for examining the mnist dataset in more detail
    """X = fetch_openml('mnist_784', version=1)
    for key in X.keys():
        value = X[key]
        print(key,type(value),"\n",value)"""    
    #return X, y
    return fetch_openml('mnist_784', version=1, return_X_y=True)
"""
def fetch():
    print("fetch")
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = np.array(y,dtype=int)
    np.savetxt("./data/der_mnist_x.csv", X,delimiter=",")
    np.savetxt("./data/der_mnist_y.csv", y,delimiter=",")
    
    np.save("./data/mnist_x.npy", X,allow_pickle=True,fix_imports=True)
    np.save("./data/mnist_y.npy", y,allow_pickle=True,fix_imports=True)
    return X, y    """
    
def randomize_data(X, y, seed=7, num_train=10000, normalize=True):
    print("randomize_data")
    # Create generator object with seed (for consistent testing across compilation)
    #gnrtr = np.random.default_rng(7)
    np.random.seed(seed)

    # Create random array with values permuted from the num elements of y
    #r = gnrtr.permutation(len(y))
    r = np.random.permutation(len(y))

    # Reorganize X and y based on the random permutation, all columns
    X, y = X[r, :], y[r]
    
    if normalize:
        max_val = np.amax(X)
        min_val = np.amin(X)
        range_val = max_val - min_val
        print(max_val, min_val, range_val)
        X = np.divide(X,range_val)

    # Assign the first num_train rows from X
    X_s, y_s, X_test, y_test = X[:num_train, :], y[:num_train], X[:num_train, :], y[:num_train]

    return X, y, X_s, y_s, X_test, y_test

def exercise_1():
    print("mnist")
    global X
    global y
    global X_s
    global y_s
    X, y, X_s, y_s = randomize_data(*load_data(),num_train=1000, normalize=True)
#    X_test, y_test = X[]
    
    """# num data to use from the full set
    train_samples = 60000
    
    # num data to use for testing
    test_size = 10000
    
    # split downloaded data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=test_size)
       """
       
    gscv5 = grid_search_SVC(X_s, y_s, SVC, 5, {'kernel':['rbf'], 
                   'C':np.linspace(.1,10000,10),
                   'gamma':np.linspace(.1,1000,10)})
    gscv5.predict()
        
        
# Regrssion using GridSearchCV
def grid_search_SVC(X, y, cclass, cv, params,refit=True):   
    gscv = GridSearchCV(cclass(), params, cv = cv, refit=refit)
    gscv.fit(X,y)
    print(str(abs(gscv.best_score_))+","+str(gscv.best_params_))
    return (gscv)
    
