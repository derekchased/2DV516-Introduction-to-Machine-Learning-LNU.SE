#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 09:30:21 2020

@author: derek
"""

import numpy as np

def extended_matrix(X):
    return np.c_[np.ones((len(X),1)),X]

def extended_matrix_deg(X,deg=1):
    assert deg >= 1 # if degree is less than 1, throw an error
    _c = np.ones((len(X),1))
    for i in range(deg):
        _c = np.c_[_c,X**(i+1)]    
    return _c

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