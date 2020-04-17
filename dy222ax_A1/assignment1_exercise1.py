#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import assignment1_knn_functions as knn_funcs


# Create first plot of original data to compare against
def exercise1_1():
    print ("\nExercise 1.1")
    
    csv_data = np.loadtxt("microchips.csv", delimiter=',') # load csv
    Xtrain = csv_data[:,0:2]; # first two columns are the data points
    Ytrain = csv_data[:,2]; # third column is the classification
    
    plt.figure(figsize=(5,5),num=1) # num markers/ticks/steps on the graph [0,1]
    plt.scatter(Xtrain[Ytrain>0,0], # plot first col on x 
                Xtrain[Ytrain>0,1], # plot second col on y
                c='g', # use the classes (0 or 1) as plot colors
                label="OK",
                s=30,
                edgecolors='' # optional, add a border to the dots
                ) 
    plt.scatter(Xtrain[Ytrain==0,0], # plot first col on x 
                Xtrain[Ytrain==0,1], # plot second col on y
                c='r', # use the classes (0 or 1) as plot colors
                label="Fail",
                s=30,
                edgecolors='' # optional, add a border to the dots
                ) 
    plt.legend()
    plt.title("Ex 1.1, Microchips OK/Fail") # gives the figure a title on top
    plt.show()
    print ("See plot.")
           
def exercise1_2():    
    print ("\nExercise 1.2")
    
    csv_data = np.loadtxt("microchips.csv", delimiter=',') # load csv
    Xtrain = csv_data[:,0:2]; # first two columns are the data points
    Ytrain = csv_data[:,2]; # third column is the classification
    Zxpredict = np.array( [[-.3,1.0], 
                       [-.5,-.1], 
                       [.6,0.0]
                       ])

    sorted_distances_indeces = knn_funcs.train(Xtrain, Zxpredict)
    kZypredict = knn_funcs.get_knn_list(range(1,8,2), Ytrain, sorted_distances_indeces)
    for i,k in enumerate(kZypredict):
        print("k = "+str(k["k"]))
        kzy = k["Zy"]
        for j,zx in enumerate(Zxpredict):
            formattedchip = np.array2string(zx,separator=", ",formatter={'float_kind':lambda x: "%.1f" % x})
            print("\tchip"+str(i),formattedchip,"==>","Fail" if kzy[j]==0 else " OK")
        
        
def exercise1_3():
    print("\nExercise 1.3")
    
    csv_data = np.loadtxt("microchips.csv", delimiter=',') # load csv
    Xtrain = csv_data[:,0:2]; # first two columns are the data points
    Ytrain = csv_data[:,2]; # third column is the classification
    
    # setup decisioun boundary vars
    h = 0.01 # Stepsize
    x1_range = np.arange(Xtrain[:,0].min()-0.1, Xtrain[:,0].max()+0.1, h) # Considered x1-values
    x2_range = np.arange(Xtrain[:,1].min()-0.1, Xtrain[:,1].max()+0.1, h) # Considered x2-values
    xx, yy = np.meshgrid(x1_range, x2_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()] # (m,2)-shaped matrix; each row is a grid point.
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Ex 1.3, Decision Boundary', fontsize=14)
    plt.style.use('default')
    fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])
    
    # compute distances once, outside of for loop
    sorted_distances_indeces = knn_funcs.train(Xtrain, Xtrain)
    sorted_distances_indeces_grid = knn_funcs.train(Xtrain, grid_points)
    
    # compute mesh, training error, and plot on figure for values of k, [1,3,5,7]
    for i, k in enumerate(range(1,8,2)):
        num_errors = (np.equal(   knn_funcs.get_knn(k, Ytrain, sorted_distances_indeces ),Ytrain) == False).sum()
        #error_rate = round((np.equal(get_knn(k, Ytrain, sorted_distances_indeces ),Ytrain) == False).sum()/Ytrain.size,3)
        
        bini=format(i, '#04b')[2:]
        axi = int(bini[0:1])
        axj = int(bini[1:2])
        #print("a", i, bin(i)[2:],format(i, '#04b')[2:3], format(i, '#04b')[3:4])
        #print("b", i, bini, axi, axj)
        
        boundary_class = knn_funcs.get_knn(k, Ytrain, sorted_distances_indeces_grid )
        clz_mesh = np.array(boundary_class).reshape(xx.shape)
        axs[axi][axj].pcolormesh(xx, yy, clz_mesh)
        axs[axi][axj].contourf(xx, yy, clz_mesh) # Alternative to pcolormesh


        axs[axi][axj].scatter(Xtrain[Ytrain>0,0], # plot first col on x 
                Xtrain[Ytrain>0,1], # plot second col on y
                c='1', # use the classes (0 or 1) as plot colors
                label="OK",
                s=30,
                edgecolors='r' # optional, add a border to the dots
                ) 
        axs[axi][axj].scatter(Xtrain[Ytrain==0,0], # plot first col on x 
                Xtrain[Ytrain==0,1], # plot second col on y
                c='0', # use the classes (0 or 1) as plot colors
                label="Fail",
                s=30,
                edgecolors='r' # optional, add a border to the dots
                )
        axs[axi][axj].legend(loc='upper right')
        axs[axi][axj].set_title("K="+str(int(k))+", Training Error: "+str(num_errors))
        
        #plt.scatter(X[:, 0], X[:, 1], c = y, marker = 'o', edgecolor = 'black')    
    plt.show()
    print("See plot.")
