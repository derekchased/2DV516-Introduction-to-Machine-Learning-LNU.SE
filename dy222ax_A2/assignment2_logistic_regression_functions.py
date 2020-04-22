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
import assignment2_matrix_functions as amf
#import traceback

#np.set_printoptions(30)
#np.seterr("raise")

# Sigmoid function for any matrix
def sigmoid_matrix(x):
    return 1.0 / (1 + np.exp(-x))

# Sigmoid function of g(Xb)
# return a probabilty between 0 and 1
def sigmoid(Xe,b):
    
    #print("sigmoid",np.dot(Xe,b))
    return sigmoid_matrix(np.dot(Xe,b))

# return the sum of all elements in the vector
# X should be extended or extended and normalized
# beta should be based on extended matrix
def cost_function(Xe, beta, y):
    return -1/Xe.shape[0]*(y.T@np.log(sigmoid(Xe,beta)) + np.transpose(1-y)@ np.log(1 - sigmoid(Xe,beta)))


# return the beta value
# X should be extended and normalized
# beta should be based on extended+normalized matrix
def gradient_descent(Xe_n, y, alpha=.005, n=1000, get_all_betas=False):    
    # Step 1 - Set an initial beta. 
    # Fill it with zeros using the num cols
    beta = np.zeros(Xe_n.shape[1])
    
    # Step 2 - Store betas in list
    betas = [beta]
    
    # Step 2 - Iterate over n, set beta to it's newest value
    #betas = [beta]
    #print(beta - alpha * np.dot(Xe_n.T,(sigmoid(Xe_n,beta) - y)))
    for i in range(1,n):
        # missing divide by n???
        beta = beta - alpha * np.dot(Xe_n.T,(sigmoid(Xe_n,beta) - y))
        
        # add new beta to list
        betas.append(beta)
    if get_all_betas: 
        return betas
    else:
        return beta

def predict(to_predict, X, b):
    all_data = np.append(X,to_predict,0)
    all_data_normalized = amf.feature_normalization(all_data)
    to_pred_normalized = np.array(  [all_data_normalized[-1]]   )
    return sigmoid(amf.extended_matrix(to_pred_normalized),b)[-1]


# print("Training errors: ",(np.sum(yy!=pp)))
# Notice: reshape(-1,1) turns an array of shape (say) (100,) 
# into shape (100,1) required by my sigmoid implementation
# logReg.sigmoid( p ).
def get_errors(Xe_n,beta,y):
    # Compute training errors
    p = np.dot(Xe_n, beta).reshape(-1,1)
    p = sigmoid_matrix( p ) # Probabilities in range [0,1] pp = np.round(p)
    pp = np.round(p)
    yy = y.reshape(-1,1)
    return np.sum(yy != pp)
    