#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:44:23 2020

@author: derek
"""

import numpy as np
import matplotlib.pyplot as plt
import assignment2_logistic_regression_functions as lorf
import assignment2_matrix_functions as amf
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def load_data():
    # import data
    data = np.loadtxt("./A2_datasets_2020/microchips.csv",delimiter=',') # load csv
    X = data[:,0:2]
    y = data[:,-1]
    return X,y
    

def exercise5_1():
    """
    Use Logistic regression and mapFeatures from the previous exercise to 
    construct nine different classifiers, one for each of the degrees 
    d ∈ [1, 9], and produce a figure containing a 3 × 3 pattern of subplots 
    showing the corresponding decision boundaries. Make sure that you pass 
    the argument C=10000
    """
    print ("\nExercise 5.1")
        

X, Y = load_data()
X1 = X[:,0]
X2 = X[:,1]
degree = 2

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




# Create Logistic Regression object
logreg = LogisticRegression(C=1000)

# Extend training data to second degree (Do not use ones)
Xe = amf.extended_matrix_deg(X,2,False)
# Train LR object using our training 2nd degree data
logreg.fit(Xe, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

"""


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X2F[:, 0], X2F[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Microchip X')
plt.ylabel('Microchip Y')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
"""

"""
fig, ax1 = plt.subplots(1,1)
fig.suptitle('Scikitlearn', fontsize=14)
fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])

X1 = X[:,0]
X2 = X[:,1]
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
plt.show()"""