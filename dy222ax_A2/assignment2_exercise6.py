#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:06:31 2020

@author: derek
"""

import numpy as np
import assignment2_matrix_functions as amf
import assignment2_linear_regression_functions as lirf
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    # Step 1 - Load Data
    Csv_data = np.loadtxt("./A2_datasets_2020/GPUBenchmark.csv",delimiter=',') # load csv
    
    # Step 2 - Setup X and y
    X = Csv_data[:,:-1]
    y = Csv_data[:,6]
    
    # Step 3 - Normalize Data
    Xn = amf.feature_normalization(X)
    
    # Step 4 - Return normalized X and labels
    return Xn, y


"""
    Forward selection using breadth first recursion
    - I had not used recursion in a while, and decided to implement it here
    - Compare is performed on a single model X. It does not
    recurse over all Mi elements
    - It finds the best performing column and returns that column and it's 
    index
"""
def compare(X, y, j=0):
    #   When you reach the last column, return the j index and the cost at j
    if j >= X.shape[1]-1:
     #   print("if",j)
        return j, cost(X,y,j)
    else:
        #   Store the index and cost of i+1 best attribute
        jplusone, jplusone_cost = compare(X,y,j+1) 
        
        #   Calculate the cost of the jurrent j
        j_cost = cost(X,y,j)
        
        #   Compare the current j and j+1, return whichever has a smaller cost
        if j_cost <= jplusone_cost:
            return j, j_cost
        else:
            return jplusone, jplusone_cost

"""
    This Cost function is used in conjunction with the compare function above
    as part of a recursive algorithm to find the best column.
    - It gets the cost using the normal equation of a single column
"""
def cost(X,y,j):
    
    # 1 - Extract the J column
    Xreduced = X[:,j].reshape(-1,1)
    
    #2 - Extend Xn
    Xe = amf.extended_matrix(Xreduced)
        
    # 3 - Get betas using normal equation
    betas = lirf.normal_equation(Xe,y)
    
    # 4 - Get Cost
    normal_eq_cost = lirf.cost_function(Xe,betas,y)

    return normal_eq_cost

"""
    This compare_M function finds the best columns according to the
    forward selection algorithm.
    - It creates a list and appends increasingly larger models, where the
    first model is the best single column, the second model are the best two
    columns, etc
    --> for example [ [col j3], [col j3, col j2], [col j3, col j2, col j4]
    - It creates a second list which keeps track of the original indeces from
    X, for each model Mi
    --> for example [ [j3], [j3, j2], [j3, j2, j4]
"""
def compare_M(X, y,k=1):
    # if k is less than 1, throw an error
    assert k>0
    
    # Create Xremain, which we will begin to slice columns out of
    Xremain = X
    
    # Create list M, which we will store each model Mi
    M=[]
    
    # Create list orig_indices, so we can keep track of how each model was
    # built
    orig_indices = []

    for i in range(k):
        
        # get the index of the best column, and get it's cost
        ind,cost = compare(Xremain,y)
        
        # extract this best column
        Xnext = Xremain[:,ind].reshape(-1,1)
        
        # add the extracted column to the new best matrix
        try:
            Xbest = np.c_[Xbest,Xnext]
        except:
            Xbest = Xnext
        
        # Extend the best matrix, this is Mi
        Xe = amf.extended_matrix(Xbest)
        
        # Get betas using normal equation of Mi
        betas = lirf.normal_equation(Xe,y)
        
        # Get Cost of Mi
        normal_eq_cost = lirf.cost_function(Xe,betas,y)
        
        # Get the original index of the extracted column
        orig_index =  np.where(Xnext == X)[1][0]
        
        # Add the index to the originals list
        orig_indices.append(orig_index)
        
        # Append the Mi model to the Models list. Attach some meta data for 
        # later use
        M.append({"model":Xbest})
        
        # Remove the extracted column from the Xremain matrix
        Xremain = np.c_[ Xremain[:,0:ind], Xremain[:,ind+1:X.shape[1] ]]
    
    return (orig_indices,M)

def exercise6_1():
    print ("\nExercise 6.1")
    print("There is no output requested for this exercise. See Ex 6.2")

def exercise6_2():
    print ("\nExercise 6.2")
    # 1 - Load Data    
    Xn, y = load_data()
    
    # 2 - Get the Models according to the forward selection algorithm
    #orig_indices, M = compare_M(Xn,y,Xn.shape[1])
    
    # 3 - Create a Kfold cross validator with 3 folds
    kf = KFold(n_splits=3)
    
    # 4 Create list to store stats on predictions and determine best model 
    best_predictions_per_k = []
    
    # 5 - Iterate over K folds
    for k,(train_index, test_index) in enumerate(kf.split(Xn)):
        #print("\nFold",k)
        # Create new list for this k value
        current_best_predictions = []
        best_predictions_per_k.append(current_best_predictions)
        
        
        # Get train and test data from the Kfold object
        X_train, X_test = Xn[train_index], Xn[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 2 - Get the Models according to the forward selection algorithm
        # from the train data
        orig_indices, M  = compare_M(X_train,y_train,X_train.shape[1])
        
        # Use the orig_indeces from ind to extract the best column from the X_test data
        Xnext = X_test[:,orig_indices[0]].reshape(-1,1)
        
        # Initilaize  X_test_reduced, which we will build up with aditional models
        X_test_build_up = Xnext
        
        # Iterate over the m models from the train data which we will use
        # to make predictions and test
        for i,models_train in enumerate(M):    
            # get the current model
            Mi = models_train["model"]
           
            # Create linear regression object
            regr = linear_model.LinearRegression()
            
            # Train the regression model using the training sets
            regr.fit(Mi, y_train)
            
            # Make predictions using the testing set
            y_pred = regr.predict(X_test_build_up)
            
            # Add MSE to predictions matrix for use later
            MSE = mean_squared_error(y_test, y_pred)
            current_best_predictions.append(MSE)
            
            # Get the next best X_test column according to our training data
            # and add it to the X_test model that we are building up
            try:
                Xnext = X_test[:,orig_indices[i+1]].reshape(-1,1)
                X_test_build_up = np.c_[X_test_build_up,Xnext]
            except:
                pass
    
    # Convert list of predictions into a matrix of size KxM  (K=3, M=6)
    np_best = np.array(best_predictions_per_k) 
    
    # Sum over each column K to get the to sum of each Mki, or, the sum of each
    # model i from any K fold
    sum_best = np.sum(np_best,0)
    
    # Get the average value of each Model
    avg_best = sum_best/np_best.shape[0]
    
    # Get the best model by finding the minimum average MSE
    min_mse_model = np.argmin(avg_best)
    
    print("The best model is M"+str(min_mse_model),
          "with a minimal avg MSE val of",
          avg_best[min_mse_model],"from 3 K folds")