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
import numpy as np

def exercise1():
    print("\nExercise B - Logistic Regression")
    pass



# Step 1 - Load Data
Csv_data = np.loadtxt("./A2_datasets_2020/admission.csv",delimiter=',') # load csv

X = Csv_data[:,0:2]
y = Csv_data[:,-1]

# Step 2 - Normalize Data
Xn = amf.feature_normalization(X)

# Step 3 - Plot data
fig, ax1 = plt.subplots(1,1)
fig.suptitle('Ex B, Logistic Regression', fontsize=14)
fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])

ax1.scatter(X[y>0,0], # plot first col on x 
                X[y>0,1], # plot second col on y
                c='1', # use the classes (0 or 1) as plot colors
                label="Admitted",
                s=30,
                edgecolors='r' # optional, add a border to the dots
                ) 
ax1.scatter(X[y==0,0], # plot first col on x 
        X[y==0,1], # plot second col on y
        c='0', # use the classes (0 or 1) as plot colors
        label="Not Admitted",
        s=30,
        edgecolors='r' # optional, add a border to the dots
        )
ax1.legend(loc='upper right')
plt.show()




csv_data = np.loadtxt("./A2_datasets_2020/admission.csv", delimiter=',') # load csv
X = np.array([[0,1],[2,3]])
y = csv_data[:,2]; # third column is the classification
beta = np.zeros((X.shape[1], 1))



#X = csv_data[:,0:2]; # first two columns are the data points

#Xe = extended_matrix(X)
#Xe_n = extended_matrix(feature_normalization(X))
#lc = logistic_cost_function(Xe,beta,y)
#lgd = logistic_gradient_descent(extended_matrix(feature_normalization(X)),y)
lcf = lorf.logistic_cost_function(amf.extended_matrix(X),beta,y)




