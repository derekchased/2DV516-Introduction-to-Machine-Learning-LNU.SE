#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""
import numpy as np

def normal_equation(Xe, y):
    beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
    return beta

def cost_function(Xe, beta, y):
    j = np.dot(Xe,beta)-y
    J = (j.T.dot(j))/len(y)
    return J

def feature_normalization(X):
    # compute mean and stdev over axis 0, the feature vector (down the column)
    mean = np.mean(X,0)
    stddev = np.std(X,0)
    
    # elementwise difference
    diff = np.subtract(X,mean)
    
    # elementwise division
    normalized = np.divide(diff,stddev)
    
    # for each feature, stddev should be 1 and mean should be 0
    #print("stddev of normalized", np.std(normalized,0))    
    #print("mean of normalized", np.mean(normalized,0))    
    
    # add vector of 1's in the first column
    Xe = extended_matrix(normalized)
    
    return Xe    

def gradient_descent(N_iterations=10, Learning_rate_alpha=.00001, B_0=(0,0)):
    pass

def extended_matrix(X):
    #return np.c_[np.ones((len(X))),X]#   is this ok? seems same without the 1
    return np.c_[np.ones((len(X),1)),X]
    





csv_data = np.loadtxt("./A2_datasets_2020/girls_height.csv") # load csv
X_parent_heights = csv_data[:,1:3]
#mom_heights = csv_data[:,1:2]
#dad_heights = csv_data[:,2:3]
y_girl_heights = csv_data[:,0:1]; # first column is the girl's height

Xe_feature_normalization = feature_normalization(X_parent_heights)
Beta_normal_equation = normal_equation(Xe_feature_normalization,y_girl_heights)
J_cost_function = cost_function(Xe_feature_normalization,Beta_normal_equation,y_girl_heights)
gradient_descent()
