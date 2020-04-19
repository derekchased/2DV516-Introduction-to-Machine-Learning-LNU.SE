#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 09:30:21 2020

@author: derek
"""

import numpy as np

# extend for 1 feature, degree of 1 only
def extended_matrix(X):
    return np.c_[np.ones((len(X),1)),X]

# extend for 1 feature, degree of 1 or more
def extended_matrix_deg(X,deg=1):
    assert deg >= 1 # if degree is less than 1, throw an error
    _c = np.ones((len(X),1))
    for i in range(deg):
        _c = np.c_[_c,X**(i+1)]    
    return _c

# extend for 2 features, any degree more than 1
def mapFeature(X1,X2,deg): # Pyton
    assert deg > 1 # if degree is less than 1, throw an error
    one = np.ones([len(X1),1])
    Xe = np.c_[one,X1,X2] # Start with [1,X1,X2]
    for i in range(2,deg+1):
        print(i)
        for j in range(0,i+1):
            Xnew = X1**(i-j)*X2**j # type (N)
            Xnew = Xnew.reshape(-1,1) # type (N,1) required by append 
            Xe = np.append(Xe,Xnew,1) # axis = 1 ==> append column
    return Xe

def feature_normalization(X):
    # compute mean and stdev over axis 0, the feature vector (down the column)
    mean = np.mean(X,0)
    stddev = np.std(X,0)
    
    # elementwise difference
    diff = np.subtract(X,mean)
    
    # elementwise division
    normalized = np.divide(diff,stddev)
    
    # for testing
    # for each feature, stddev should be 1 and mean should be 0
    #print("stddev of normalized", np.std(normalized,0))    
    #print("mean of normalized", np.mean(normalized,0))    
    
    return normalized  