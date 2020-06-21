#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""
import numpy as np
from collections import Counter # for counting class of neighbors

def get_distances_optimal1(X_train, Zxpredict):
    """ Get the distances between Xtrain and Zpredict
    Note:
        1. Optimization level - highly optimized!
        
        Algorithm was found https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        It is a no loop solution, which means it can handle the matrices in
        one line of code rather than having to iterate over the rows of one
        matrix
        
        2. I have removed the square root to further optimize the calculation.
        This is ok, because I do not need the actual distance between points. 
        I just need a list of the relative distances. Since the sq root is 
        monotonic, it preserverses the order.         
    """
    #dists = np.sqrt(-2 * np.dot(X, X_train.T) + np.sum(X_train**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis])
    dists = -2 * np.dot(Zxpredict, X_train.T) + np.sum(X_train**2, axis=1) + np.sum(Zxpredict**2, axis=1)[:, np.newaxis]
    return dists

"""def get_distances_optimal2(X_train, Z):
    "" Get the distances between Xtrain and Zpredict
    Note:
        Optimization level - medium optimized! 
    ""
    dists = []
    for z in Z:
        dists.append(np.sum((z-X_train)**2, axis= 1))
    return dists"""

def get_knn(k, Ytrain, sorted_distances_indeces):
    # create list to store classifactions of Zx
    Zypredict = []
    
    # iterate through each list of distances of Z and X points
    for dist in sorted_distances_indeces:    
        
        # reduce list to first k items, then use the indeces to get the classifications from Y        
        k_nearest_neighbors_of_Z_0 = [  Ytrain[i] for i in dist[:k]]
        
        # obtain the mode from the nearest neighbors
        mode = Counter( k_nearest_neighbors_of_Z_0).most_common()[0][0] 
        
        # store the mode for this Z
        Zypredict.append( mode )
    
    return Zypredict

def get_knn_list(k_list, Ytrain, sorted_distances_indeces):
    # create list to store each k iteration
    kZypredict = []
    
    # loop through each k value requested
    for k in k_list:
        
        # store the classifications of this Z into the list of K
        kZypredict.append({"k":k,"Zy":get_knn(k, Ytrain, sorted_distances_indeces)})
    
    return kZypredict
    
def train(Xtrain, Zxpredict):
    # get the distances between each Z and all X data points
    distances = get_distances_optimal1(Xtrain, Zxpredict)
    
    # sort the distances using argsort to retrieve the corresponding index
    sorted_distances_indeces = np.argsort(distances) 
    
    return sorted_distances_indeces

def train_regression(Xtrain):
    # get the distances between each Z and all X data points
    distances = get_distances_optimal1(Xtrain, Xtrain)
    
    return distances[0]

def get_knn_regression(k, Ytrain, sorted_distances_indeces):
    # create list to store classifactions of Zx
    Zypredict = []
    
    # iterate through each list of distances of Z and X points
    for dist in sorted_distances_indeces:    
        
        # reduce list to first k items, then use the indeces to get the classifications from Y        
        k_nearest_neighbors_of_Z_0 = [  Ytrain[i] for i in dist[:k]]
        
        # obtain the mode from the nearest neighbors
        mean = k_nearest_neighbors_of_Z_0.sum()/k
        
        # store the mode for this Z
        Zypredict.append( mean )
    
    return Zypredict
