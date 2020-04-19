#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import assignment2_logistic_regression_functions as lorf
import assignment2_matrix_functions as amf
from matplotlib.colors import ListedColormap

# Load Data
Csv_data = np.loadtxt("./A2_datasets_2020/admission.csv",delimiter=',') # load csv
X = Csv_data[:,0:2]
y = Csv_data[:,-1]
Xe = amf.extended_matrix(X)
Xe_n = amf.extended_matrix(amf.feature_normalization(X))


def exercise1():
    print("\nExercise B - Logistic Regression")
    
    # Ex B.1
    
    # Normalize Data
    Xn = amf.feature_normalization(X)
    
    # Plot data
    fig, ax1 = plt.subplots(1,1)
    fig.suptitle('Ex B.1, Logistic Regression (normalized data)', fontsize=14)
    fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
    
    ax1.scatter(Xn[y>0,0], # plot first col on x 
                    Xn[y>0,1], # plot second col on y
                    c='1', # use the classes (0 or 1) as plot colors
                    label="Admitted",
                    s=30,
                    edgecolors='r' # optional, add a border to the dots
                    ) 
    ax1.scatter(Xn[y==0,0], # plot first col on x 
            Xn[y==0,1], # plot second col on y
            c='0', # use the classes (0 or 1) as plot colors
            label="Not Admitted",
            s=30,
            edgecolors='r' # optional, add a border to the dots
            )
    ax1.legend(loc='upper right')
    plt.show()
    
    
    # Ex B.2
    
    print( "\nB.2, sigmoid",lorf.sigmoid_matrix(np.array([[0,1],[2,3]])))
    
    # Ex B.3
    
    print( "\nB.3, Xe",amf.extended_matrix(X))
    
    # Ex B.4
    
    
    beta = np.zeros(Xe_n.shape[1])
    lcf = lorf.cost_function(Xe_n, beta, y)
       
    print("\nB.4, logistic cost, beta=[0,0,0]::", lcf, "\n(solution [0,0,0] / .6931)")
    
    # Ex B.5
    global lgd
    lgd = lorf.gradient_descent(
              amf.extended_matrix(
                  amf.feature_normalization(X)),y,.005,1)
    print("\nB.5, gradient_descent alpha=.005 n=1::beta=", lgd, 
          "cost=",lorf.cost_function(Xe_n, lgd, y),
          "\n(solution B1=[.05,0.141,0.125] / J=.6217)")
    
    # Ex B.6    
    
    lgd = lorf.gradient_descent(
          amf.extended_matrix(
              amf.feature_normalization(X)),y,.005,1000)
    print("\nB.6, gradient_descent alpha=.005 n=1000::beta=", lgd, 
          "cost=",lorf.cost_function(Xe_n, lgd, y),
          "\n(solution Bn=[1.686,3.923,3.657] / J=.2035)")
    
    
    
    
    
    # Plot also the linear decision boundary
    Xn = amf.feature_normalization(X)
    X1 = Xn[:,0]
    X2 = Xn[:,1]
    #Xe_n2 = amf.mapFeature(Xn[:,0], Xn[:,1], 2)
    h=.01 # stepsize in the mesh
    x_min, x_max = X1.min()-0.1, X1.max()+0.1
    y_min, y_max = X2.min()-0.1, X2.max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) # Mesh Grid 
    x1,x2 = xx.ravel(), yy.ravel() # Turn to two Nx1 arrays
    XXe = amf.mapFeature(x1,x2,2) # Extend matrix for degree 2
    lgd2 = lorf.gradient_descent(amf.mapFeature(X1,X2,2),y,.005,1000)
    p = lorf.sigmoid( XXe, lgd2  )  # classify mesh ==> probabilities
    classes = p>0.5 # round off probabilities
    clz_mesh = classes.reshape(xx.shape) # return to mesh format
    
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) # mesh plot  
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) # colors
    
    plt.figure(2)
    plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light) 
    plt.scatter(X1,X2,c=y, marker=".", cmap=cmap_bold) 
    plt.show()
    
    
    # Ex B.7    
    predict = lorf.predict(np.array([[45,85]]), X, lgd)
    
    # Compute training errors
    errors = lorf.get_errors(Xe_n, lgd, y)
    print("\nB.7, predict [45 85]::", predict,"training errors::",errors,
          "\n(solution predict=.77, errors=11")
