#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:49:50 2020

@author: derek

Example from, https://github.com/rafaelmessias/2dv516/blob/mas2dv516-python-example-decision-boundaries.ipynbter/2dv516-python-example-decision-boundaries.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import assignment1_knn_functions as knn_funcs

def get_polynomial_data():
    data = np.loadtxt("polynomial200.csv", delimiter=',')
    Xtrain = data[:100,:];
    ytest = data[100:200,:];
    return Xtrain, ytest    

def exercise2_1():
    print ("\nExercise 2.1")
    Xtrain, ytest = get_polynomial_data()
    print ("Loads data. See exercise 2.2 for plot showing data.")

def exercise2_2():
    print ("\nExercise 2.2")
    Xtrain, ytest = get_polynomial_data()
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    fig.suptitle('Ex 2.2, Side-by-Side', fontsize=14)
    ax1.scatter(Xtrain[:,0],Xtrain[:,1],c='#40925a',marker='1',label="Training Data")
    ax2.scatter(ytest[:,0],ytest[:,1],c='#e82d8f',marker='.',label="Testing Data")
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('f(x)')
    plt.show()
    print("See plot.")
    
def exercise2_3():
    """ For the regression, I was not able to finish this exercise. 
    - I understand the concept that regression is for a continuous function
    rather than a classification. 
    - I understand that we the nearest neighbors based on the X coordinate,
    take the corresponding Y values of the k nearest neighbors, proivde the
    mean and that becomes the y output of f(x)
    - However, I was not able to translate this conecpt into code. I will
    continue to study and work with the TA's to achieve full familiarty
    - In my solution, I have trained the data to itself, and then calculate
    the regression. So for k=1, the line fits all the points. 
    """
    
    
    print ("\nExercise 2.3")
    
    # load data
    data = np.loadtxt("polynomial200.csv", delimiter=',')
    Xtrain = data[:100,:];
    ytest = data[100:200,:];
    
    # sort by X
    Xtrain_sorted_by_x = Xtrain[Xtrain[:,0].argsort()]
    
    # interval, and create steps
    #x_min, x_max = Xtrain[:,0].min(), Xtrain[:,0].max()
    #h = (x_max-x_min)/10000
    #xrange = np.arange(x_min, x_max, h)
    
    for k in range(1,8,2):
        # train data
        sorted_distances_indeces = knn_funcs.train(Xtrain_sorted_by_x, Xtrain_sorted_by_x)
        
        # extract the first column
        xt1= Xtrain_sorted_by_x[:,1]
        
        # list of f(x) predictions
        Zypredict = []
            
        # iterate through each list of distances of Z and X points
        for dist in sorted_distances_indeces:    
            
            # reduce list to first k items, then use the indeces to get the classifications from Y        
            k_nearest_neighbors_of_Z_0 = np.array([  xt1[i] for i in dist[:k]])
            
            # obtain the mean from the nearest neighbors
            mean = k_nearest_neighbors_of_Z_0.sum()/k
            
            # store the mode for this Z
            Zypredict.append( mean )
        
        # Calc MSE from actual Y values and f(x) values
        MSE = np.sum((Xtrain_sorted_by_x[:,1]-Zypredict)**2)/len(Zypredict)
        
        # plot the regression
        plt.figure(figsize=(5,5),num=1) # num markers/ticks/steps on the graph [0,1]
        plt.plot(Xtrain_sorted_by_x[:,0], # plot first col on x 
                    Zypredict, # plot second col on y
                    c='g', # use the classes (0 or 1) as plot colors
                    ) 
        plt.scatter(Xtrain_sorted_by_x[:,0], Xtrain_sorted_by_x[:,1])
        #plt.legend()
        plt.title("polynomial_train, k = "+str(k)+", MSE = " + str(MSE)) # gives the figure a title on top
        plt.show()
        
        print("MSE =", MSE )
        
    print ("See plots")