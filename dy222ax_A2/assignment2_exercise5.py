#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:44:23 2020

@author: derek
"""

import numpy as np
import matplotlib.pyplot as plt
import assignment2_matrix_functions as amf
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def load_data():
    # import data
    data = np.loadtxt("./A2_datasets_2020/microchips.csv",delimiter=',') # load csv
    X = data[:,0:2]
    y = data[:,-1]
    return X,y

def exercise5_1_and_2():
    # load data
    X, y = load_data()
    
    # Separate vectors
    X1 = X[:,0]
    X2 = X[:,1]
    
    Cs = [10000,1]
    
    # Iterate over the C values request in 5.1 and 5.2
    for i,C in enumerate(Cs):
        # Create figure
        fig = plt.figure()
        fig.suptitle("Ex 5."+str(i) +', C='+str(C), fontsize=12)
        fig.tight_layout(pad=1.0, rect=[0, 0.03, 10, 0.95])
        
        for degree in range(1,10):
            # Create subplot for this degree
            ax = fig.add_subplot(3, 3, degree)
            
            # Create extended matrix of degree
            Xe = amf.mapFeature(X1,X2,degree,ones=False) # No 1-column!
            
            # Create LogisticRegression object with C hyperparameter
            logreg = LogisticRegression(C=C, tol=1e-6,max_iter=10000)
            
            # fit the model with data
            logreg.fit(Xe,y)

            # Create meshgrid
            h=.01
            x_min, x_max = X1.min()-0.1, X1.max()+0.1
            y_min, y_max = X2.min()-0.1, X2.max()+0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h)) # Mesh Grid 
            x1,x2 = xx.ravel(), yy.ravel() # Turn to two Nx1 arrays
            XXe = amf.mapFeature(x1,x2,degree,False)
            
            # predict using the meshgrid's ravel
            y_pred=logreg.predict(XXe) 
            
            # Label as True of False
            classes = y_pred>0.5
            
            # Return to mesh format
            clz_mesh = classes.reshape(xx.shape) 
            
            # Create colors
            cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]) # mesh plot  
            cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"]) # colors
            
            # Show the boundary
            ax.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light) 
            
            # Show the data points
            ax.scatter(X1,X2,c=y, marker=".", cmap=cmap_bold) 

            # compute the accuracy of this C value and show on the plot
            # we will compare each degree/C pair to see which C value
            # produced a more accurate model
            ax.set_title("Deg "+str(degree)+", Acc. "+
                         str(int(round(logreg.score(Xe,y),2)*100))+"%")
                      
        # Show the grid
        plt.subplots_adjust(hspace = .65)
        plt.subplots_adjust(top=.88)
        plt.show()

    print ("\nExercise 5.1 - See plot")
    print ("\nExercise 5.2  - See plot")
    print("The C value determines the strenth of the regularization penalty. ",
          "A lower value of C will strengthen the regularization. ",
          "This is because it has an inverse relationship, so smaller ",
          "values increase strength while larger values decrease strength. ",
          "\n In this case, we can see that C=1 produced a tighter decision ",
          "boundary while C=10000 produced a more liberal boundary. ",
          "I have also indicated the accuracy of each model. We can see that ",
          "the C=10000 has a higher accuracy when compared degree to degree. ",
          "This matches the more encompassing decision boundary for C=10000. ",
          "However, it is most likely a poorer model for most degrees in ",
          "general case. This is because, it is most likely overfit. In ",
          "exercise 5.3, we will see see which C value is best for each ",
          "polynomial.")

# Regrssion using GridSearchCV
def regression_grid_search_cvs():    
    
    # load data
    X, y = load_data()
    
    # Separate vectors
    X1 = X[:,0]
    X2 = X[:,1]
    
    print("Performing a Grid Search to find the best hyperparameters ",
          "penalty of 'l2' or 'none' and C value in the range ",
          "np.linspace(1.0,10000.0,100). The findings are displayed below ",
          "in text rathr than as a plot. The regularized model almost ",
          "always performed better across all degrees. However, degree 2 ",
          "preferred no penalty.")
    for degree in range(1,10):
        # Create extended matrix of degree
        Xe = amf.mapFeature(X1,X2,degree,ones=False) # No 1-column!
        # For Elastic Net, add array of l1_ratio values to params
        params = {"penalty":["l2","none"],"C":np.linspace(1.0,10000.0,100)}#np.linspace(1.0,10000,1000)}
        
        regressor = LogisticRegression()
        
        gscv = GridSearchCV(regressor, params, cv = 5)
        
        gscv.fit(Xe,y)
        
        print("Degree " + str(degree) + ", Optimal MSE="+str(abs(gscv.best_score_))+
              ", Optimal Params="+str(gscv.best_params_))

def exercise5_3():
    print ("\nExercise 5.3")
    regression_grid_search_cvs()