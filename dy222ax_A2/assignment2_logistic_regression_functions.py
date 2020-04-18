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

np.set_printoptions(30)
np.seterr("raise")

# LOGISTIC REGRESSION
# g(z)
# I am getting a vector rather than a scalar
def sigmoid_matrix(x):
    return np.divide(1,(1 + (np.exp(np.negative(x)))))

# g(Xb)
# return a probabilty between 0 and 1
def sigmoid(Xe,b):
    return 1 / (1 + np.exp( -np.dot(Xe,b)))

# return the sum of all elements in the vector
def logistic_cost_function(Xe, beta, y):
    print("logistic_cost_function")
    print(Xe.shape)
    print(beta.shape)
    return np.sum(-1/Xe.shape[0]*
                  (y.T*np.log(sigmoid(Xe,beta)) + 
                   np.transpose(1-y)*
                   np.log(1 - sigmoid(Xe,beta))))
    

def logistic_b_next(Xe_n, y, beta, alpha,n):
    return beta - (alpha/n) * np.dot(Xe_n.T, sigmoid(Xe_n, beta) - y)

def logistic_gradient_descent(Xe_n, y, alpha=.005, n=1):
    # Step 1 - Set an initial beta. Fill it with zeros (as suggested in lecture video)
    # Use size of num of vectors/columns
    beta = np.zeros(np.ma.size(Xe_n,1))
    
    # Step 2 - Initilaize b[0]
    betas = [logistic_b_next(Xe_n, y, beta, alpha,n)]
    
    # Step 3 - Iterate over n
    for j in range(1,n):
        print(j)
        # use try/except to prevent adding inf and similar errors to betas
        try:
            bj = logistic_b_next(Xe_n, y, betas[j-1], alpha, n)
        except:
            #print("Error\n", traceback.print_exc())
            break
        # do not add inf values or repeat values. if so, halt and return
        # in other words, no need to complete all n iterations
        if np.isfinite(bj).all() and not np.all(np.equal(bj,betas[j-1])):
            betas.append(bj)
        else:
            break
    
    return np.array(betas)

