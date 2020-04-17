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
    Xtest = data[100:200,:];
    return Xtrain, Xtest    

def exercise2_1():
    print ("\nExercise 2.1")
    Xtrain, Xtest = get_polynomial_data()
    print ("Loads data. See exercise 2.2 for plot showing data.")

def exercise2_2():
    print ("\nExercise 2.2")
    Xtrain, Xtest = get_polynomial_data()
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    fig.suptitle('Ex 2.2, Side-by-Side', fontsize=14)
    ax1.scatter(Xtrain[:,0],Xtrain[:,1],c='#40925a',marker='1',label="Training Data")
    ax2.scatter(Xtest[:,0],Xtest[:,1],c='#e82d8f',marker='.',label="Testing Data")
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('f(x)')
    plt.show()
    print("See plot 2.2")
    
def exercise2_3_and_4():
    print ("\nExercise 2.3")
    print("See plots.")
    print ("\nExercise 2.4")
    
    # load data
    Xtrain, Xtest = get_polynomial_data()
    
    # sort by X (for readability)
    Xtrain_sorted_by_x = Xtrain[Xtrain[:,0].argsort()]
    
    # interval, and create steps
    #x_min, x_max = Xtrain[:,0].min(), Xtrain[:,0].max()
    #h = (x_max-x_min)/10000
    #xrange = np.arange(x_min, x_max, h)
    
    for k in range(1,8,2):
        # train data using first columns
        sorted_distances_indeces = knn_funcs.train(
            Xtrain_sorted_by_x[:,0].reshape(-1, 1), 
            Xtrain_sorted_by_x[:,0].reshape(-1, 1))
        
        #predict Xtest
        sorted_distances_indeces_test = knn_funcs.train(
            Xtrain_sorted_by_x[:,0].reshape(-1, 1),
            Xtest[:,0].reshape(-1, 1)) 
        
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
        plt.title("Ex 2.3, polynomial_train, k = "+str(k)+", MSE = " + str(MSE)) # gives the figure a title on top
        plt.show()
        
        #print("\nk= "+str(k),"\nTRAIN MSE =", MSE )
        
        # list of f(x) predictions
        y_pred = []
        
        # predict the test here
        for dist in sorted_distances_indeces_test:
            # reduce list to first k items, then use the indeces to get the classifications from Y        
            k_nearest_neighbors_of_Z_0 = np.array([  xt1[i] for i in dist[:k]])
            # obtain the mean from the nearest neighbors
            mean = k_nearest_neighbors_of_Z_0.sum()/k
            # store the mode for this Z
            y_pred.append( mean )
        MSE = np.sum(( Xtest[:,1] - y_pred )**2)/len(y_pred) #calculate the MSE for the test
        print("K=",k," TEST MSE: %s" % (MSE))

def exercise2_5():
    print("\nExercise 2.5")
    print("\tThe best value of K is generally the one that has the minmum "+
          "MSE value. There are some caveats, such as concerns with scale "+
          "and performance and whether a 'more expensive' evaluation at some "+
          "k is worth the extra overhead if it only has slightly better results"+
          "In this case, we have chosen just a few small K values 1 through "+
          "7, and our data set is not so large. So in this case, the k "+
          "value 5 has the smallest MSE of 28.5. Therefore for this "+
          "exercise, K=5 is the best choice.\n\tIf performance were an issue, "+
          "for this example, we might be satisfied with K=3, which has a "+
          "similar MSE value of 31.6. This MSE might be suitable for us "+
          "depending on the circumstances.")