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



"""
import matplotlib.pyplot as plt
import assignment2_logistic_regression_functions as lorf
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
"""
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
    

def exercise6_1():
    print ("\nExercise 6.1")

def cost(X,y,j):
    Xreduced = X[:,j].reshape(-1,1)
    #print(Xreduced)
    #print(Xreduced.shape)
    #2 - Extend Xn
    Xe = amf.extended_matrix(Xreduced)
        
    # 3 - Get betas using normal equation
    betas = lirf.normal_equation(Xe,y)
    
    # 4 - Get Cost
    normal_eq_cost = lirf.cost_function(Xe,betas,y)
    #print("COST:: j\t",j,"j_cost\t",normal_eq_cost)
    return normal_eq_cost

    
#   Forward selection using breadth first recursion
def compare(X, y, j=0):
    #print("compare",j)
    #   End when you've got to the last column. Return the index and the cost
    if j >= X.shape[1]-1:
     #   print("if",j)
        return j, cost(X,y,j)
    else:
        #   Recurse over j+1... Will return the best feature from J+1 -> p
        jplusone, jplusone_cost = compare(X,y,j+1) 
        
        #   Calculate the current cost
        j_cost = cost(X,y,j)
        
        print("j1\t",jplusone," j+1C\t",jplusone_cost)
        print("j\t",j,"jC\t",j_cost)
        
        #   Compare j cost to the best j+1 cost, return whichever is best
        if j_cost <= jplusone_cost:
            return j, j_cost
        else:
            return jplusone, jplusone_cost

# 1 - Load Data    
Xn, y = load_data()
besti,bestcost = compare(Xn,y)
print(besti,bestcost,Xn[:,besti],"\n\n")


def compare_M(X, y,k=1):
    assert k>0
    Xremain = X
    M=[]
    orig_indices = []

    for i in range(k):
        #print("k",i)
        #print("Xremain",Xremain.shape,Xremain)
        ind,cost = compare(Xremain,y)
        #print(i,"ind",ind,"cost",cost)
        #print("Xremain[:,ind]",Xremain[:,ind])
        Xnext = Xremain[:,ind].reshape(-1,1)
        try:
            Xbest = np.c_[Xbest,Xnext]
        except:
            Xbest = Xnext
        
        #print(Xnext == X)
        #print("ORIG INDEX", np.where(Xnext == X)[1])

        # Extend matrix        
        Xe = amf.extended_matrix(Xbest)
        
        # 3 - Get betas using normal equation
        betas = lirf.normal_equation(Xe,y)
        
        # 4 - Get Cost
        normal_eq_cost = lirf.cost_function(Xe,betas,y)
        
        orig_index =  np.where(Xnext == X)[1][0]
        orig_indices.append(orig_index)
        
        M.append({"ind":i,"model_cost":normal_eq_cost,
                  "original_index_of_added_feature":orig_index,
                  "betas":betas,"model":Xbest})
        
        Xremain = np.c_[ Xremain[:,0:ind], Xremain[:,ind+1:X.shape[1] ]]
        #print("\n\n",k,"==Xremain==\n",Xremain.shape,Xremain)
    return (orig_indices,M)
orig_indices, M = compare_M(Xn,y,Xn.shape[1])

#for m in M:
    #print(m)
    
    
    
kf = KFold(n_splits=3)
#print(kf.get_n_splits(Xn))
#print(kf)

for ind, (train_index, test_index) in enumerate(kf.split(Xn)):
    """print("\nFold:",ind)
    print("TRAIN:", train_index, "TEST:", test_index)"""
    X_train, X_test = Xn[train_index], Xn[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(X_train,"\n", X_test)
    #print(y_train,"\n", y_test)
    orig_indices, M  = compare_M(X_train,y_train,X_train.shape[1])
    
    print("\nK Fold =",ind)
    #print("orig_indices",orig_indices)
    
    #print("X_test_reduced",X_test_reduced)
    Xnext = X_test[:,orig_indices[0]].reshape(-1,1)
    X_test_reduced = Xnext
    
    for innerind,m in enumerate(M):    
        print("\nModel =",innerind, ":: features =",orig_indices[0:innerind+1])
        #print("model",m["model"])
        #print("y_train",y_train)
        #print("X_test",X_test)
        model = m["model"]
        
        #print(X_test_reduced)

        
        # Create linear regression object
        regr = linear_model.LinearRegression()
        
        # Train the model using the training sets
        regr.fit(model, y_train)
        
        
        
        
            
        #print("hello",X_test_reduced)
        
        # Make predictions using the testing set
        y_pred = regr.predict(X_test_reduced)
        
        # The coefficients
        #print('Coefficients: \n', regr.coef_)
        # The mean squared error
        print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))
        print(y_test)
        print(np.round(y_pred,1))
        
        try:
            Xnext = X_test[:,orig_indices[innerind+1]].reshape(-1,1)
            X_test_reduced = np.c_[X_test_reduced,Xnext]
        except:
            pass
        
        
        
        