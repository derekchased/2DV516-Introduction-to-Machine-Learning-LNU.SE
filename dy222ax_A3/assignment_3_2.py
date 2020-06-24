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

def ex_1(num_train = 12000,C=[.1,10,1000],gamma=[.1,10,1000]):
    print("A3, Ex2")
    X, y = load_data()
    X = as3f.normalize_mnist_data(X)
    X, y, X_s, y_s = as3f.randomize_and_split_data(X, y, num_train=num_train)
    svc_param = [{'kernel':['rbf'], 'C':C,'gamma':gamma}]

    gscv = as3f.grid_search_SVC(X_s, y_s, SVC, 5, svc_param)

    
def one_vs_all():
    pass


def load_data():
    print("load_data")
    return fetch_openml('mnist_784', version=1, return_X_y=True)
    
