#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:37:53 2020

@author: derek
"""

from sklearn.model_selection import GridSearchCV
import numpy as np

# Regrssion using GridSearchCV
def grid_search_SVC(X, y, cclass, cv, params,refit=True,print_score=True):   
    if print_score:
        print("grid_search_SVC",params)
    gscv = GridSearchCV(cclass(), params, cv = cv, refit=refit,n_jobs=-1)
    gscv.fit(X,y)
    if print_score:
        print(str(abs(gscv.best_score_))+","+str(gscv.best_params_))
    return (gscv)

def randomize_data(X, y, seed=7, num_train=637):
    print("randomize_data",num_train)
    # Create generator object with seed (for consistent testing across compilation)
    #gnrtr = np.random.default_rng(7)
    np.random.seed(seed)

    # Create random array with values permuted from the num elements of y
    #r = gnrtr.permutation(len(y))
    r = np.random.permutation(len(y))

    # Reorganize X and y based on the random permutation, all columns
    X, y = X[r, :], y[r]
    
    # Assign the first 5000 rows from X
    X_s, y_s = X[:num_train, :], y[:num_train]

    return X, y, X_s, y_s

def normalize_mnist_data(X):
    print("normalize mnist data")
    max_val = np.amax(X)
    min_val = np.amin(X)
    range_val = max_val - min_val
    return np.divide(X,range_val)

def print_gscv(gscv):
    bc = gscv.cv_results_
    be = gscv.best_estimator_
    bs = gscv.best_score_
    bp = gscv.best_params_
    bi = gscv.best_index_
    bss = gscv.scorer_
    bn = gscv.n_splits_
    br = gscv.refit_time_
    print(bc,be,bs,bp,bi,bss,bn,br)