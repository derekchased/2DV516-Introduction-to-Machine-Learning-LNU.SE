#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:58:14 2020

@author: derek
"""
import numpy as np
import matplotlib.pyplot as plt
import assignment2_logistic_regression_functions as lorf
import assignment2_matrix_functions as amf
from matplotlib.colors import ListedColormap

def load_data():
    # import data
    data = np.loadtxt("./A2_datasets_2020/microchips.csv",delimiter=',') # load csv
    X = data[:,0:2]
    y = data[:,-1]
    return X,y
    

def exercise4_1():
    print ("\nExercise 4.1")
    Xn, y = load_data()
    
    # Plot data
    fig, ax1 = plt.subplots(1,1)
    fig.suptitle('Ex 4.1, Microchips', fontsize=14)
    fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
    
    # Use two different scatters I want the legend to show the different labels
    
    ax1.scatter(Xn[y>0,0], # plot first col on x 
                    Xn[y>0,1], # plot second col on y
                    c='1', # use the classes (0 or 1) as plot colors
                    label="OK",
                    s=30,
                    edgecolors='r' # optional, add a border to the dots
                    ) 
    ax1.scatter(Xn[y==0,0], # plot first col on x 
            Xn[y==0,1], # plot second col on y
            c='0', # use the classes (0 or 1) as plot colors
            label="Fail",
            s=30,
            edgecolors='r' # optional, add a border to the dots
            )
    ax1.legend(loc='upper right')
    plt.show()
    
def exercise4_2():
    print ("\nExercise 4.2")
    Xn, y = load_data()
    
    # Set feature vars
    X1 = Xn[:,0]
    X2 = Xn[:,1]
    degree = 2
    Xe = amf.mapFeature(X1,X2,degree)
    alpha = .12905
    n = 1000
    betas = lorf.gradient_descent(Xe,y,alpha,n,True)
    costs = []
    for i,beta in enumerate(betas):
        costs.append([i,lorf.cost_function(Xe, beta, y)])
    beta = betas[-1]
    
    # Compute training errors
    errors = lorf.get_errors(Xe, beta, y)
    correct = len(Xe)-errors
    accuracy = (correct-errors)/len(Xe)
    
    # Create 1x2 plot
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Ex 4.2, Deg2, Microchips N='+str(n)+", Alpha="+str(alpha), fontsize=14)
    fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])
    
    # Plot the costs on the first figure
    c0 = np.array(costs)[:,0]
    c1 = np.array(costs)[:,1]
    ax1.plot(c0,c1)
    ax1.set_xlabel("N Iterations")
    ax1.set_ylabel("Cost")
    
    # 4.2.2 Decision Boundary
    h=.01 # stepsize in the mesh
    x_min, x_max = X1.min()-0.1, X1.max()+0.1
    y_min, y_max = X2.min()-0.1, X2.max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) # Mesh Grid 
    x1,x2 = xx.ravel(), yy.ravel() # Turn to two Nx1 arrays
    XXe = amf.mapFeature(x1,x2,degree) # Extend matrix for degree 2
    
    p = lorf.sigmoid( XXe, beta  )  # classify mesh ==> probabilities
    classes = p>0.5 # round off probabilities
    clz_mesh = classes.reshape(xx.shape) # return to mesh format
    
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) # mesh plot  
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) # colors
    
    ax2.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light) 
    ax2.scatter(X1,X2,c=y, marker=".", cmap=cmap_bold) 
    ax2.set_title("Training errors ="+str(errors))
    plt.show()
        
def exercise4_4():
    print ("\nExercise 4.4")
    Xn, y = load_data()
    
    # Set feature vars
    X1 = Xn[:,0]
    X2 = Xn[:,1]
    degree = 5
    Xe = amf.mapFeature(X1,X2,degree)
    alpha = .15
    n = 100000
    #Alpha = 0.105 N = 100000 Cost = 0.2964006494452656
    #Alpha = 0.15 N = 100000  Cost = 0.2962931749771741
    betas = lorf.gradient_descent(Xe,y,alpha,n,True)
    costs = []
    for i,beta in enumerate(betas):
        costs.append([i,lorf.cost_function(Xe, beta, y)])
    print ("Alpha =",alpha,"N =",n,"Cost =",costs[-1][1])
    beta = betas[-1]
    
    # Compute training errors
    errors = lorf.get_errors(Xe, beta, y)
    correct = len(Xe)-errors
    accuracy = (correct-errors)/len(Xe)
    #print("[Train] Errors::",errors, "Correct::",correct,"Accuracy::",accuracy)
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Ex 4.4, Deg 5, Microchips N='+str(n)+", Alpha="+str(alpha), fontsize=14)
    fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])
    
    c0 = np.array(costs)[:,0]
    c1 = np.array(costs)[:,1]
    
    ax1.plot(c0,c1)
    ax1.set_xlabel("N Iterations")
    ax1.set_ylabel("Cost")
    
    
    # 4.2.2 Decision Boundary
    h=.01 # stepsize in the mesh
    x_min, x_max = X1.min()-0.1, X1.max()+0.1
    y_min, y_max = X2.min()-0.1, X2.max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) # Mesh Grid 
    x1,x2 = xx.ravel(), yy.ravel() # Turn to two Nx1 arrays
    XXe = amf.mapFeature(x1,x2,degree) # Extend matrix for degree 2
    
    p = lorf.sigmoid( XXe, beta  )  # classify mesh ==> probabilities
    classes = p>0.5 # round off probabilities
    clz_mesh = classes.reshape(xx.shape) # return to mesh format
    
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) # mesh plot  
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) # colors
    
    ax2.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light) 
    ax2.scatter(X1,X2,c=y, marker=".", cmap=cmap_bold) 
    ax2.set_title("Training errors ="+str(errors))
    plt.show()

    
exercise4_1()
exercise4_2()
exercise4_4()
