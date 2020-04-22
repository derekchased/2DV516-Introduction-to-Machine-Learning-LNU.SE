#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:34:09 2020

@author: derek
"""

import numpy as np
from sklearn.model_selection import train_test_split
import assignment2_matrix_functions as amf
import assignment2_logistic_regression_functions as lorf
import matplotlib.pyplot as plt


def exercise3_1():
    
    print ("\nExercise 3")
    
    # Ex 3.1 Part A
    
    # import data
    data = np.loadtxt("./A2_datasets_2020/breast_cancer.csv",delimiter=',') # load csv
    X = data[:,0:9]
    y = data[:,-1] # benign = 2, malignant = 4
    
    
    # Ex 3.1 Part B
    
    # Split data - 
    # @NOTE - Using this method was approved on Slack by TA
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    
    # Ex 3.2
    
    # Modify labels
    for i in range(len(y_train)): 
        y_train[i] = 0 if y_train[i] == 2 else 1
        
    for i in range(len(y_test)): 
        y_test[i] = 0 if y_test[i] == 2 else 1
    
    # Ex 3.3
    
    X_train_extended_normalized = amf.extended_matrix(amf.feature_normalization(X_train))
    alpha, n = .0001,10000
    betas = lorf.gradient_descent(X_train_extended_normalized,y_train,alpha,n,get_all_betas=True)
    costs =[]
    for i, beta in enumerate(betas):
        costs.append([i,lorf.cost_function(X_train_extended_normalized, beta, y_train)])
    
    # Plot data
    fig, ax1 = plt.subplots(1,1)
    fig.suptitle('Ex 3.3, Linear Logistic Regression Cost - α = '+str(alpha)+' n = '+str(n), fontsize=14)
    fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
    c0 = np.array(costs)[:,0]
    c1 = np.array(costs)[:,1]
    
    ax1.plot(c0,c1)
    ax1.set_xlabel("N Iterations")
    ax1.set_ylabel("Cost")
    plt.show()
    
    #print("\nEx 3.3 - See plot")
    
    # Ex 3.4
    
    # Compute training errors
    errors = lorf.get_errors(X_train_extended_normalized, beta, y_train)
    correct = len(X_train_extended_normalized)-errors
    accuracy = (correct-errors)/len(X_train_extended_normalized)
    
    print("\nEx 3.4")
    print("Train:: Errors =",errors, "Correct =",correct,"Accuracy =",accuracy)
    
    # Ex 3.5
    
    # Compute training errors
    X_test_extended_normalized = amf.extended_matrix(amf.feature_normalization(X_test))
    test_errors = lorf.get_errors(X_test_extended_normalized, beta, y_test)
    test_correct = len(X_test_extended_normalized)-test_errors
    test_accuracy = (test_correct-test_errors)/len(X_test_extended_normalized)
    print("\nEx 3.5")
    print("Test:: Errors =",test_errors, "Correct =",test_correct,"Accuracy =",test_accuracy)
    


def exercise3_2():
    print ("\nEx 3.2")
    print("I split the data into two groups. The test data has 20% of the values",
          "and the training data has 80% of the values. I chose this because",
          "training the data with as much data as possible is the most ",
          "important part of training a model. With increased training data, ",
          "we will see a decrease in the number of errors on the training data.",
          " Further, the 80/20 split seems to be",
          "a good split for an average data set, as a rule of thumb. With ",
          "large data sets, we can begin to move more data from training ",
          "and into test data if needed.")

def exercise3_6():
    print("\nEx 3.6")
    print("Shuffling the data will give us some noise, but they are ",
          "essentially the same results within a margin. On repeated ",
          "shuffles of the data, we can see that the results will change by 5-10% ",
          "upwards or downards. This gives us a good idea of the upper and lower ",
          "bounds of our model for similar data.\nThe amount of data set aside ",
          "for training vs testing will always affect the outcome, whether it ",
          "is shuffled or not. The more training data, the lower the testing errors.")
    
    
    
    