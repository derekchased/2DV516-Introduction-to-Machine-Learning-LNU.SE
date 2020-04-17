#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import assignment2_linear_regression_functions as alrf

# Create first plot of original data to compare against
def exerciseA_1():
    print ("\nExercise A.1")

    # Load Data
    csv_data = np.loadtxt("./A2_datasets_2020/girls_height.csv") # load csv
    Heights_y_girl = csv_data[:,0] # first column is the girl's height
    Heights_X_parent = csv_data[:,1:3]
    Heights_X_mom = csv_data[:,1]
    Heights_X_dad = csv_data[:,2]

    # A1.1 - Plot Data
    fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,sharex=True)
    fig.suptitle('Ex A.1, Girl Height in inches', fontsize=14)
    ax1.set(xlabel="Mom Height",ylabel="Girl Height")
    ax2.set(xlabel="Dad Height")
    ax1.scatter(Heights_X_mom,Heights_y_girl,c='#e82d8f',marker='1')
    ax2.scatter(Heights_X_dad,Heights_y_girl,c='#40925a',marker='2')
    plt.show()
    
    # A1.2 - Compute Extended Matrix
    Xe_parents = alrf.extended_matrix(Heights_X_parent)
    print("Extended Matrix of Parent's Heights\n",Xe_parents,"\n")
    
    # A1.3 - Compute Normal Equation and Make a Prediction
    Beta_normal_parents = alrf.normal_equation(Xe_parents,Heights_y_girl)
    pred_ex = alrf.extended_matrix(np.array([[65,70]]))
    y_parents_normal_eq = alrf.predict(pred_ex,Beta_normal_parents)
    print("==> Prediction of girl height with parental heights of 65,70\n", 
          y_parents_normal_eq[0],"\n")
    
    # A1.4 - Apply Feature Normalization, plot dataset, 
    # heights should be centered around 0 with a standard deviation of 1.
    global X_feature_normalized_heights
    X_feature_normalized_heights = alrf.feature_normalization(Heights_X_parent)
    fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,sharex=True)
    fig.suptitle('Ex A.1, Girl Height in inches', fontsize=14)
    ax1.set(xlabel="Mom Height Normalized",ylabel="Girl Height")
    ax2.set(xlabel="Dad Height Normalized")
    ax1.scatter(X_feature_normalized_heights[:,0],Heights_y_girl,c='#e82d8f',marker='1')
    ax2.scatter(X_feature_normalized_heights[:,1],Heights_y_girl,c='#40925a',marker='2')
    plt.show()
    
    # A1.5 - Compute the extended matrix Xe and apply the Normal equation 
    # on the normalized version of (65.70). The prediction should 
    # still be 65.42 inches.
    Xe_feature_normalized_heights = alrf.extended_matrix(X_feature_normalized_heights)
    Beta_normal_parents_normalized = alrf.normal_equation(Xe_feature_normalized_heights,Heights_y_girl)
    heights_to_predict = np.array([[65,70]])
    Heights_plus_pred = np.append(Heights_X_parent, heights_to_predict,0)
    Normalized_heights_plus_pred = alrf.feature_normalization( Heights_plus_pred )
    Normalized_heights_to_pred = np.array(  [Normalized_heights_plus_pred[-1]]   )
    Xe_Normalized_heights_to_pred = alrf.extended_matrix(Normalized_heights_to_pred)
    y_parents_pred = alrf.predict(Xe_Normalized_heights_to_pred,Beta_normal_parents_normalized)
    print("==> Prediction of girl height with normalized parental heights of 65,70\n", 
          y_parents_pred[0],"\n")
    
    # A1.6 - Implement the cost function J(β) = n1 (Xeβ − y)T (Xeβ − y) as a 
    # function of parameters Xe,y,β. The cost for β from the Normal 
    # equation should be 4.068.
    cost_function_normalized = alrf.cost_function(Xe_feature_normalized_heights,Beta_normal_parents_normalized,Heights_y_girl)
    print("==> Cost Function (normalized)\n",cost_function_normalized,"\n")
    
    cost_function = alrf.cost_function(Xe_parents,
                                       Beta_normal_parents,Heights_y_girl)
    print("==> Cost Function not-normalized\n",cost_function,"\n")
    
    
def exerciseA_1_gradient():
    print ("\nExercise A.1 Gradient")
    # A1.7 - Gradient descent βj+1 = βj − αXeT (Xeβj − y).
    # (a) Implement a vectorized version of gradient descent
    # (b) Find (and print) suitable hyperparameters α,N. Remember to start 
    # with a small α (say 0.001) and N (say 10) to make sure that the cost is 
    # decreasing. A plot J vs Itera- tions is an excellent way to see that 
    # J is decrasing as expected. Then gradually decrease α, and increase N, 
    # to find a suitable pair of (α,N) that rapidly decreases/stabilizes the 
    # cost at its minimum (4.068).
    # (c) Verify that the predicted height for a girl with parents (65.70)
    # is still 65.42 inches.
    
    # Load Data
    csv_data = np.loadtxt("./A2_datasets_2020/girls_height.csv") # load csv
    Heights_y_girl = csv_data[:,0] # first column is the girl's height
    Heights_X_parent = csv_data[:,1:3]
    Heights_X_mom = csv_data[:,1]
    Heights_X_dad = csv_data[:,2]
    
    ### Gradient
    # Step 1 - Feature Normalize X
    X_fnormalized_grad = alrf.feature_normalization(Heights_X_parent)
    
    # Step 2 - Extend the normalized matrix
    Xe_grad = alrf.extended_matrix(X_fnormalized_grad)
    
    # Step 3 - Set an initial beta. Fill it with zeros. Use size of num of vectors/columns
    beta_grad_start = np.zeros(np.ma.size(Xe_grad,1))
    
    # Step 4 - Get the gradient descent array
    global beta_grad_iterations
    beta_grad_iterations = alrf.gradient_descent(Xe_grad, Heights_y_girl, beta_grad_start,alpha=.001,n=1000)
    
    # Step 5 - Take the most recent/last beta value
    beta_grad_final = beta_grad_iterations[-1]
    
    # Step 6 - Calculate cost function for each beta
    global J_gradient
    J_gradient = []
    for i,j in enumerate(beta_grad_iterations):
        J_grad = alrf.cost_function(Xe_grad,beta_grad_iterations[i],Heights_y_girl)
        J_gradient.append(J_grad)
        
    fig, ax1 = plt.subplots()
    fig.suptitle('Ex A.1 Gradient Descent, alpha = .001', fontsize=14)
    ax1.set(xlabel="Number of iterations = "+str(len(beta_grad_iterations)),ylabel="Cost J, min = "+str(round(J_gradient[-1],3)))
    ax1.plot(np.arange(0,len(beta_grad_iterations)),J_gradient)
    plt.xlim(0, len(beta_grad_iterations))
    plt.show()
    
    
    
    ### Prediction
    # Step 1 - Create prediction array
    heights_to_predict = np.array([[65,70]])
    
    # Step 1 - Add heights that you want to predict to the other heights
    Heights_plus_pred = np.append(Heights_X_parent, heights_to_predict,0)
    
    # Step 2 - Normalize the new heights matrix
    Normalized_heights_plus_pred = alrf.feature_normalization( Heights_plus_pred )
    
    # Step 3 - Extract the normalized heights that you need to predict
    Normalized_pred2 = np.array(  [Normalized_heights_plus_pred[-1]]   )
    
    # Step 4 - Extend the predictions matrix
    pred_ex_grad = alrf.extended_matrix(Normalized_pred2)
    
    # Step 5 - Predict the y
    y_parents_grad = alrf.predict(pred_ex_grad,beta_grad_final)
    print("==> The predicted height for a girl with parents (65.70) is:\n", round(y_parents_grad[0],2))


"""
# Load Data
csv_data = np.loadtxt("./A2_datasets_2020/girls_height.csv") # load csv
Heights_y_girl = csv_data[:,0] # first column is the girl's height
Heights_X_parent = csv_data[:,1:3]
Heights_X_mom = csv_data[:,1]
Heights_X_dad = csv_data[:,2]
    
# Normal Equation (Mom)
Xe_mom = extended_matrix(Heights_X_mom)
Beta_normal_mom = normal_equation(Xe_mom,Heights_y_girl)
J_mom = cost_function(Xe_mom,Beta_normal_mom,Heights_y_girl)

# Normal Equation (Dad)
Xe_dad = extended_matrix(Heights_X_dad)
Beta_normal_dad = normal_equation(Xe_dad,Heights_y_girl)
J_dad = cost_function(Xe_dad,Beta_normal_dad,Heights_y_girl)

# Normal Equation (Parents)
Xe_parents = extended_matrix(Heights_X_parent)
Beta_normal_parents = normal_equation(Xe_parents,Heights_y_girl)
J_parents = cost_function(Xe_parents,Beta_normal_parents,Heights_y_girl)

# Predict girl's height from arbitrary mom+dad heights
pred_ex = extended_matrix(np.array([[65,70]]))
y_parents_normal_eq = predict(pred_ex,Beta_normal_parents)


### Gradient
# Step 1 - Feature Normalize X
X_fnormalized_grad = alrf.feature_normalization(Heights_X_parent)

# Step 2 - Extend the normalized matrix
Xe_grad = alrf.extended_matrix(X_fnormalized_grad)

# Step 3 - Set an initial beta. Fill it with zeros. Use size of num of vectors/columns
beta_grad_start = np.zeros(np.ma.size(Xe_grad,1))

# Step 4 - Get the gradient descent array
beta_grad_iterations = alrf.gradient_descent(Xe_grad, Heights_y_girl, beta_grad_start,alpha=.001,n=1000)

# Step 5 - Take the most recent/last beta value
beta_grad_final = beta_grad_iterations[-1]

# Optional - Check on the cost function
J_grad = alrf.cost_function(Xe_grad,beta_grad_final,Heights_y_girl)

### Prediction
# Step 1 - Create prediction array
heights_to_predict = np.array([[65,70]])

# Step 1 - Add heights that you want to predict to the other heights
Heights_plus_pred = np.append(Heights_X_parent, heights_to_predict,0)

# Step 2 - Normalize the new heights matrix
Normalized_heights_plus_pred = alrf.feature_normalization( Heights_plus_pred )

# Step 3 - Extract the normalized heights that you need to predict
Normalized_pred2 = np.array(  [Normalized_heights_plus_pred[-1]]   )

# Step 4 - Extend the predictions matrix
pred_ex_grad = alrf.extended_matrix(Normalized_pred2)

# Step 5 - Predict the y
y_parents_grad = alrf.predict(pred_ex_grad,beta_grad_final)
"""