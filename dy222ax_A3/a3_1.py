#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:14:35 2020

@author: derek
"""

from sklearn.svm import SVC
import numpy as np
import plt_functions as pltf
import a3_funcs as as3f
import matplotlib.pyplot as plt


def load_data():
    data = np.loadtxt('./data/mnistsub.csv',delimiter=',')
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y

def exercise_1():
    print("A3 Ex1 and Ex2")
    X, y = load_data()
    X = as3f.normalize_mnist_data(X)
    X, y, X_train, y_train, X_test, y_test = as3f.randomize_and_split_data(X, y)
    
    # Hypertuning values of C and gamma
    test_params = [.1,1,10,100,1000]
    
    # Setup each model
    svc_params = [{'kernel':['linear'],'C':test_params},
                  {'kernel':['rbf'], 'C':test_params,'gamma':test_params},
                  {'kernel':['poly'], 'C':test_params,'degree':[2,3,4,5,6]}]
    
    # Loop through each model
    for svc_param in svc_params:
        # CV and Regression using Grid Search Cross Validation
        gscv = as3f.grid_search_SVC(X_train, y_train, SVC, 5, svc_param)
        
        # Analysis and plot of each model
        exercise_1_2(gscv, X_train, y_train, X_test, y_test )
    
    
def exercise_1_2(gscv, X_train, y_train, X_test, y_test):
    # Get the classifier object
    clf = gscv.best_estimator_
    
    
    # Separate vectors
    X1 = X_train[clf.support_, 0]
    X2 = X_train[clf.support_, 1]
    
    # Meshgrid
    xx, yy = pltf.get_meshgrid(X1, X2)
    
    # plot boundary and data points for Train set
    fig = plt.figure()
    title = "Ex 1 - TRAIN accuracy "+str(round(abs(gscv.best_score_),5))+" "
    for key in gscv.best_params_:
        title = title+key+":"+str(gscv.best_params_[key])+" "
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1)
    pltf.add_countour(ax, xx, yy, clf, colors='r',linewidths=0.2)
    ax.scatter(X_train[:,0], X_train[:,1], s=.5,c=y_train)
    plt.show()
    
    # plot boundary and data points for Test set
    accuracy = str(round(abs(clf.score(X_test, y_test)),5))
    kernel = str(gscv.best_params_["kernel"])
    fig = plt.figure()
    title = "Ex 1- TEST accuracy, kernel: " + kernel + ", accuracy: " + accuracy+" "
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1)
    pltf.add_countour(ax, xx, yy, clf, colors='r',linewidths=0.2)
    ax.scatter(X_test[:,0], X_test[:,1], s=.5,c=y_test)
    plt.show()
    
exercise_1()