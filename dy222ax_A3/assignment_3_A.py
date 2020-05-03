"""
Created on 4/25/20, 3:29 PM
Author derek
Project introml
File assignment_3.py
"""

from sklearn.svm import SVC
import numpy as np
import plt_functions as pltf
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt('./data/bm.csv',delimiter=',')
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


def randomize_data(X, y):
    # Create generator object with seed (for consistent testing across compilation)
    #gnrtr = np.random.default_rng(7)
    np.random.seed(7)

    # Create random array with values permuted from the num elements of y
    #r = gnrtr.permutation(len(y))
    r = np.random.permutation(len(y))

    # Reorganize X and y based on the random permutation, all columns
    X, y = X[r, :], y[r]

    # Assign the first 5000 rows from X
    X_s, y_s = X[:5000, :], y[:5000]

    return X, y, X_s, y_s

def exercise_1():
    
    X, y, X_s, y_s = randomize_data(*load_data())
    
    # Ex A.2
    # rbf = gaussian,
    clf = SVC(kernel="rbf", gamma=.5, C=20)
    clf.fit(X_s, y_s)
    print("Ex A_2, accuracy:",clf.score(X_s, y_s))
    
    # Ex A.3
    
    # Separate vectors
    X1 = X_s[clf.support_, 0]
    X2 = X_s[clf.support_, 1]
    y_pred = y_s[clf.support_]
    
    # Meshgrid
    xx, yy = pltf.get_meshgrid(X1, X2)

    # plot boundary with support vector    
    fig = plt.figure()
    fig.suptitle("Ex A, Decision boundary with support vector")
    ax = fig.add_subplot(1, 1, 1)
    ax, Z = pltf.add_countour(ax, xx, yy, clf, colors='r')
    
    # plot support vectors
    ax.scatter(X1, X2, s=1,c=y_pred)

    # plot boundary with data    
    fig = plt.figure()
    fig.suptitle("Ex A, Decision boundary with data")
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(xx, yy, Z,colors='r')
    ax.scatter(X_s[:,0], X_s[:,1], s=.5,c=y_s)
    plt.show()
    
