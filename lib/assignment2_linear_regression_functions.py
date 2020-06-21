#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

@NOTICE
    The vectorised Cost Function, Normal Equation, and Gradient Descent are 
    identical to the case y = β1 + β2x! ⇒ A proper Python solution 
    can be reused

"""
import numpy as np
import traceback
import assignment2_matrix_functions as amf

np.set_printoptions(30)
np.seterr("raise")

# training MSE
# if zero we have perfect fit
# with respect to training data
def cost_function(Xe, beta, y):
    j = np.dot(Xe,beta)-y
    J = (j.T.dot(j))/len(y)
    return J

def normal_equation(Xe, y):
    # find betas that minimizes the cost function
    # Minimizing training the mse is not necessarily the best
    return np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

def predict(Xe, B):
    # predect y by taking dot product of extended X and betas
    # Model: y = β1 + β2X1 + β3X2, or, Vectorised Approach: Model y = Xβ
    #print("asdasd\n",Xe,"\n",B)
    return np.dot(Xe,B)

### GRADIENT DESCENT
def b_next(Xe_n, y, beta, alpha):
    return beta - alpha * np.dot( Xe_n.T, np.dot(Xe_n, beta) - y)
    
def gradient_descent(Xe_n, y, alpha=.001, n=1000):
    # Step 1 - Set an initial beta. Fill it with zeros (as suggested in lecture video)
    # Use size of num of vectors/columns
    beta = np.zeros(np.ma.size(Xe_n,1))
    
    # Step 2 - Initilaize b[0]
    betas = [b_next(Xe_n, y, beta, alpha)]
    
    # Step 3 - Iterate over n
    for j in range(1,n):
        # use try/except to prevent adding inf and similar errors to betas
        try:
            bj = b_next(Xe_n, y, betas[j-1], alpha)
        except:
            print("Error\n", traceback.print_exc())
            break
        # do not add inf values or repeat values. if so, halt and return
        # in other words, no need to complete all n iterations
        if np.isfinite(bj).all() and not np.all(np.equal(bj,betas[j-1])):
            betas.append(bj)
        else:
            break
    return betas

def predict_gradient(X, zx, betas):
    # Step 1 - Combine the predication to your other data
    X_plus_z = np.append(X, zx,0)
    
    # Step 2 - Normalize the combined data
    X_plus_z_normalized = amf.feature_normalization( X_plus_z )
    
    # Step 3 - Extract the normalized prediction data
    z_normalized = np.array(  [X_plus_z_normalized[-1]]   )
    
    # Step 4 - Predict the y
    zy = predict(amf.extended_matrix(z_normalized),betas[-1])
    
    return zy

