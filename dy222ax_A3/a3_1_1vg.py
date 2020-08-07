#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:14:35 2020

@author: derek
"""

import numpy as np
from sklearn.svm import SVC
import a3_funcs as as3f

def compute_gram_X_Y(X, Y, sigma = .01, degree = 2):
    #print("X\n",X,"Y\n",Y,\n")
    # Get num rows
    n = len(X)
    m = len(Y)

    # Where X.shape = (n,p), and Y.shape = (m,p), create an (n,m) matrix G
    G = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            xrow = X[i,:]
            yrow = Y[j,:]
            # Compute the x,y index of G using the formula
            G[i,j]= np.sum( np.exp( -sigma * (xrow-yrow) **2 )) ** degree

    return G

def load_data():
    data = np.loadtxt('./data/mnistsub.csv',delimiter=',')
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y

def exercise_1_1vg():
    print("exercise1_1vg")
    X, y = load_data()
    X = as3f.normalize_mnist_data(X)
    X, y, X_train, y_train, X_test, y_test = as3f.randomize_and_split_data(X, y)
    
    # store best values
    best_sig = None
    best_deg = None
    best_c = None
    best_accuracy = None

    for deg in [2]:#1,2,3,4,5]:
        for c in [10000]:#1,10,100,1000,10000]:
            for sig in [.01]:#.01,.1,1]:
                
                # Compute gram X_train and X_train
                G_train = compute_gram_X_Y(X_train,X_train,sig,deg)
                
                # Create SVC instance and fit using precomputed gram matrix
                clf = SVC(kernel="precomputed",C=c)
                clf.fit(G_train,y_train)
                
                # Predict using gram of X_test and X_train
                G_test = compute_gram_X_Y(X_test,X_train)
                y_pred = clf.predict(G_test)
                
                # Get accuracy to find best hyperparameters
                accuracy = np.sum(y_pred == y_test)/len(y_test)
                #print("C:",c,"Sig:",sig,"Deg:",deg)
                #print("Acc:", accuracy)
                
                # store best hyperparameters
                try:
                    if accuracy > best_accuracy:
                        best_sig = sig
                        best_deg = deg
                        best_c = c
                        best_accuracy = accuracy
                except:
                    best_sig = sig
                    best_deg = deg
                    best_c = c
                    best_accuracy = accuracy
         
    # print best hyperparameters
    print ("Best ",best_accuracy, "sig",best_sig, "deg", best_deg, "c",best_c)
