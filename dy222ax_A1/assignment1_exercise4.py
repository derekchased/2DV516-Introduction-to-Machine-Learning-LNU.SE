#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap


# Create first plot of original data to compare against
# This is the same as Exercise 1.1
def exercise4_1():
    print ("\nExercise 4.1")
    
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
    plt.title("Ex 4.1, Microchips OK/Fail") # gives the figure a title on top
    plt.show()
    print ("See plot.")
           
    
def exercise4_2():    
    print ("\nExercise 4.2")
    csv_data = np.loadtxt("microchips.csv", delimiter=',') # load csv
    Xtrain = csv_data[:,0:2]; # first two columns are the data points
    Ytrain = csv_data[:,2]; # third column is the classification
    Zxpredict = np.array( [[-.3,1.0], 
                       [-.5,-.1], 
                       [.6,0.0]
                       ])

    # loop over values of K
    for k in range(1,8,2):
        print("k = " + str(k))
        
        # Instantiate new KNC class and pass it current K value
        clf = neighbors.KNeighborsClassifier(k)
        
        # Use our training data to fit the model
        clf.fit(Xtrain, Ytrain)
        
        # Predict the classifications of the microchip
        predicted = clf.predict(Zxpredict)
        
        # For each chip, print out the data
        for i in range(len(predicted)):
            formattedchip = np.array2string(Zxpredict[i],separator=", ",formatter={'float_kind':lambda x: "%.1f" % x})
            print("\tchip"+str(i),formattedchip,"==>","Fail" if predicted[i]==0 else " OK")        
        
def exercise4_3():
    print("\nExercise 4.3")
    # This function was adapted from exercise 1.3 as well as the 
    # examples that can be found on sci-kit's website 
    
    
    csv_data = np.loadtxt("microchips.csv", delimiter=',') # load csv
    Xtrain = csv_data[:,0:2]; # first two columns are the data points
    Ytrain = csv_data[:,2]; # third column is the classification
    
    h = .01  # step size in the mesh
    
    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    
    
    # Create mesh coordinates
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create the axis subplots
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Ex 4.3, Decision Boundary', fontsize=14)
    plt.style.use('default')
    #fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])

    # Iterate over k values
    for i, k in enumerate(range(1,8,2)):
        
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(k)
        clf.fit(Xtrain, Ytrain)
        calc = clf.predict(Xtrain)
        
        # Calculate errors by checking for values that don't match and 
        # summing the total
        errors = (np.equal(calc,Ytrain) == False).sum()
        
        # Make new prediction over the mesh array
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        
        # Creative use of binary to draw the plots in 2x2 grid
        bini=format(i, '#04b')[2:]
        axi = int(bini[0:1])
        axj = int(bini[1:2])
        
        # Draw the color mesh
        axs[axi][axj].pcolormesh(xx, yy, Z, cmap=cmap_light)
    
        # Plot also the training points
        axs[axi][axj].scatter(Xtrain[Ytrain>0, 0], Xtrain[Ytrain>0, 1], c="r", cmap=cmap_bold, edgecolor='k', s=20, label="OK")
        axs[axi][axj].scatter(Xtrain[Ytrain==0, 0], Xtrain[Ytrain==0, 1], c="b", cmap=cmap_bold, edgecolor='k', s=20, label="Fail")
        axs[axi][axj].legend(loc='upper right')
        axs[axi][axj].set_title("K="+str(int(k))+", Training Error: "+str(errors))
    
    plt.show()
    print("See plot.")