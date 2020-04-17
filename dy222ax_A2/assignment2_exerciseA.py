#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import assignment2_linear_regression_functions as alrf

def load_data():
    csv_data = np.loadtxt("./A2_datasets_2020/girls_height.csv") # load csv
    Heights_y_girl = csv_data[:,0] # first column is the girl's height
    Heights_X_parent = csv_data[:,1:3]
    return Heights_X_parent, Heights_y_girl    

def exerciseA_1():
    print ("\nExercise A.1")

    # Load Data
    global X, y
    X, y = load_data()
    
    # A1.1 - Plot Data
    fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,sharex=True)
    fig.suptitle('Ex A.1, Girl Height in inches', fontsize=14)
    ax1.set(xlabel="Mom Height",ylabel="Girl Height")
    ax2.set(xlabel="Dad Height")
    ax1.scatter(X[:,0],y,c='#e82d8f',marker='1')
    ax2.scatter(X[:,1],y,c='#40925a',marker='2')
    plt.show()
    
    # A1.2 - Compute Extended Matrix
    Xe_parents = alrf.extended_matrix(X)
    print("Extended Matrix of Parent's Heights\n",Xe_parents,"\n")
    
    # A1.3 - Compute Normal Equation and Make a Prediction
    Beta_normal_parents = alrf.normal_equation(Xe_parents,y)
    y_parents_normal_eq = alrf.predict(alrf.extended_matrix(np.array([[65,70]])),Beta_normal_parents)
    print("==> Prediction of girl height with parental heights of 65,70\n", 
          y_parents_normal_eq[0],"\n")
    
    # A1.4 - Apply Feature Normalization, plot dataset, 
    # heights should be centered around 0 with a standard deviation of 1.
    X_feature_normalized_heights = alrf.feature_normalization(X)
    fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,sharex=True)
    fig.suptitle('Ex A.1, Girl Height in inches', fontsize=14)
    ax1.set(xlabel="Mom Height Normalized",ylabel="Girl Height")
    ax2.set(xlabel="Dad Height Normalized")
    ax1.scatter(X_feature_normalized_heights[:,0],y,c='#e82d8f',marker='1')
    ax2.scatter(X_feature_normalized_heights[:,1],y,c='#40925a',marker='2')
    plt.show()
    
    # A1.5 - Compute the extended matrix Xe and apply the Normal equation 
    # on the normalized version of (65.70). The prediction should 
    # still be 65.42 inches.
    Xe_feature_normalized_heights = alrf.extended_matrix(X_feature_normalized_heights)
    Beta_normal_parents_normalized = alrf.normal_equation(Xe_feature_normalized_heights,y)
    heights_to_predict = np.array([[65,70]])
    Heights_plus_pred = np.append(X, heights_to_predict,0)
    Normalized_heights_plus_pred = alrf.feature_normalization( Heights_plus_pred )
    Normalized_heights_to_pred = np.array(  [Normalized_heights_plus_pred[-1]]   )
    y_parents_pred = alrf.predict(alrf.extended_matrix(Normalized_heights_to_pred),Beta_normal_parents_normalized)
    print("==> Prediction of girl height with normalized parental heights of 65,70\n", 
          y_parents_pred[0],"\n")
    
    # A1.6 - Implement the cost function J(β) = n1 (Xeβ − y)T (Xeβ − y) as a 
    # function of parameters Xe,y,β. The cost for β from the Normal 
    # equation should be 4.068.
    cost_function_normalized = alrf.cost_function(Xe_feature_normalized_heights,Beta_normal_parents_normalized,y)
    print("==> Cost Function (normalized)\n",cost_function_normalized,"\n")
    
    cost_function = alrf.cost_function(Xe_parents,
                                       Beta_normal_parents,y)
    print("==> Cost Function not-normalized\n",cost_function,"\n")

def exerciseA_1_gradient():
    print ("\nExercise A.1 Gradient")
    
    # Load Data
    X, y = load_data() 
    
    ### Gradient
    # Step 1 - Normalize and Extend X
    Xe_normalized = alrf.extended_matrix(alrf.feature_normalization(X)) 
    
    # Step 2 - Calculate betas using gradient descent
    betas = alrf.gradient_descent2(Xe_normalized, y, alpha=.001,n=1000)
    
    # Step 3 - Calculate cost function for each beta
    J_gradient = []
    for i,j in enumerate(betas):
        J_grad = alrf.cost_function(Xe_normalized,betas[i],y)
        J_gradient.append(J_grad)
    
    # Step 4 - Plot the cost over iterations
    fig, ax1 = plt.subplots()
    fig.suptitle('Ex A.1 Gradient Descent, alpha = .001', fontsize=14)
    ax1.set(xlabel="Number of iterations = "+str(len(betas)),ylabel="Cost J, min = "+str(round(J_gradient[-1],3)))
    ax1.plot(np.arange(0,len(betas)),J_gradient)
    plt.xlim(0, len(betas))
    plt.show()
    
    # Step 5 - Predict arbitrary height
    
    # 5a) Place in matrix
    heights_to_predict = np.array([[65,70]])
    
    # 5b) Place in matrix
    y_parents_grad = alrf.predict_gradient(X,heights_to_predict, betas)
    
    print("==> The predicted height for a girl with parents (65.70) is:\n", round(y_parents_grad[0],2))