#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import assignment2_linear_regression_functions as lirf
import assignment2_matrix_functions as amf

def exercise1_1():
    print ("\nExercise 1 - Normal Equation")
    # Step 1 - Load Data
    Csv_data = np.loadtxt("./A2_datasets_2020/GPUBenchmark.csv",delimiter=',') # load csv
    
    X = Csv_data[:,:-1]
    y = Csv_data[:,6]
    
    # Step 2 - Normalize Data
    Xn = amf.feature_normalization(X)
    
    # Step 3 - Plot data
    fig, ax = plt.subplots(2,3)
    fig.suptitle('Ex 1.1, Multivariate Data Sets', fontsize=14)
    fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
    titles = ["CudaCores","BaseClock","BoostClock","MemorySpeed",
              "MemoryConfig","MemoryBandwidth","BenchmarkSpeed"]
    
    # iterate over columns of Xn by using the Transpose of Xn
    i, j = 0,0
    for ind, xi in enumerate(Xn.T):
        ax[i][j].scatter(xi,y)
        ax[i][j].set_title(titles[ind])
        #ax[i][j].set_xlim([xi.min()-1.5, xi.max()+1.5])
        j +=1
        if j==3: i,j = 1,0
    plt.show()
    
    # Step 4 - Get extended matrix
    Xe = amf.extended_matrix(X)
    
    # Step 5 - Get betas using normal equation
    betas = lirf.normal_equation(Xe,y)
    
    # Step 6 - Create prediction matrix
    pred = np.array([[2432, 1607,1683,8, 8, 256]])
    
    # Step 7 - Make prediction
    y_pred = lirf.predict(amf.extended_matrix(pred),betas)[0]
    print("Predicted benchmark:", y_pred," \tActual benchmark: 114",)
    
    # Step 9 - What is the cost J(β) when using the β computed by 
    # the normal equation above?
    normal_eq_cost = lirf.cost_function(Xe,betas,y)
    print("Cost:",normal_eq_cost)
    
    print ("\nExercise 1 - Gradient Descent")
    # Gradient - Step 1 - Normalize and Extend X
    Xe_n = amf.extended_matrix(amf.feature_normalization(X)) 
    
    # Step 2 - Calculate betas using gradient descent
    alpha, n = .01, 1000
    betas = lirf.gradient_descent(Xe_n, y, alpha,n)
    
    # Step 3 - Calculate cost function for each beta
    J_gradient = []
    for i,j in enumerate(betas):
        J_grad = lirf.cost_function(Xe_n,betas[i],y)
        J_gradient.append(J_grad)
        
    grad_cost = J_gradient[-1] 
    print("alpha =",str(alpha)," n =", str(n))
    print("Cost:",str(grad_cost))
    print("Gradient cost within",str(  round(100*abs(grad_cost-normal_eq_cost)/normal_eq_cost,5) )+"% of normal cost -> This is less than 1%!")
    
    # Step XXX - Predict benchmark
    y_parents_grad = lirf.predict_gradient(X,np.array([[2432, 1607,1683,8, 8, 256]]), betas)
    
    print("Predicted benchmark:",y_parents_grad[0])