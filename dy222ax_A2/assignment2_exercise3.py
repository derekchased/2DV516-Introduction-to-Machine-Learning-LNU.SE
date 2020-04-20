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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
    # Ex 3.2
    
    # Modify labels
    for i in range(len(y_train)): 
        y_train[i] = 0 if y_train[i] == 2 else 1
        
    for i in range(len(y_test)): 
        y_test[i] = 0 if y_test[i] == 2 else 1
    
    print("I split the data into two groups. The test data has 20% of the values",
          "and the training data has 80% of the values. I chose this because")
    
    
    # Ex 3.3
    
    """
    3. Normalize the training data and train a linear logistic regression 
    model using gradient descent. Print the hyperparameters α and N iter and 
    plot the cost function J(β) as a function over iterations.
    """
    
    X_train_extended_normalized = amf.extended_matrix(amf.feature_normalization(X_train))
    alpha, n = .0001,10000
    betas = lorf.gradient_descent(X_train_extended_normalized,y_train,alpha,n,get_all_betas=True)
    costs =[]
    for i, beta in enumerate(betas):
        costs.append([i,lorf.cost_function(X_train_extended_normalized, beta, y_train)])
    
    print("\n\nbeta:",beta,"\ncost:", lorf.cost_function(X_train_extended_normalized, beta, y_train))
    
    
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
    
    
    # Ex 3.4
    """What is the training error (number of non-correct classifications 
    in the training data) and the training accuracy
    (percentage of correct classifications) for your model?"""
    
    # Compute training errors
    errors = lorf.get_errors(X_train_extended_normalized, beta, y_train)
    correct = len(X_train_extended_normalized)-errors
    accuracy = (correct-errors)/len(X_train_extended_normalized)
    print("[Train] Errors::",errors, "Correct::",correct,"Accuracy::",accuracy)
    
    # Ex 3.5
    """What is the number of test error and the test accuracy for your model?"""
    
    # Compute training errors
    X_test_extended_normalized = amf.extended_matrix(amf.feature_normalization(X_test))
    test_errors = lorf.get_errors(X_test_extended_normalized, beta, y_test)
    test_correct = len(X_test_extended_normalized)-test_errors
    test_accuracy = (test_correct-test_errors)/len(X_test_extended_normalized)
    print("[Test] Errors::",test_errors, "Correct:: ",test_correct,"Accuracy::",test_accuracy)
    
