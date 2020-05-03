#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:54:26 2020

@author: derek
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
import assignment_3_funcs as as3f

def exercise_1():
    print("A3, Ex2")
    X, y, X_s, y_s = as3f.randomize_data(*load_data(),num_train=1000, normalize=True)

    gscv5 = as3f.grid_search_SVC(X_s, y_s, SVC, 5, {'kernel':['rbf'], 
                   'C':np.linspace(.1,10000,10),
                   'gamma':np.linspace(.1,1000,10)})
    gscv5.predict()
    
def one_vs_all():
    pass


def load_data():
    print("load_data")
    return fetch_openml('mnist_784', version=1, return_X_y=True)
    
X, y = load_data()
X = as3f.normalize_mnist_data(X)
X, y, X_s, y_s = as3f.randomize_data(X, y, num_train=500)
test_params = [.1,1,10,100,1000,10000]
svc_param = [{'kernel':['rbf'], 'C':test_params,'gamma':test_params}]

gscv = as3f.grid_search_SVC(X_s, y_s, SVC, 5, svc_param)
