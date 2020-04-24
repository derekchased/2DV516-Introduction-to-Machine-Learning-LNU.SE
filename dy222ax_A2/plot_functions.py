#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:10:52 2020

@author: derek
"""

import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('seaborn-white')


def plot_grid(X,Y,sharex=False,sharey=False,show=True):    
    print(X)
    #i,j = X.shape
    
    fig, ax = plt.subplots(3, 3,sharex=sharex, sharey=sharey)
    # axes are in a two-dimensional array, indexed by [row, col]
    for iind in range(3):
        for jind in range(3):
            #print(X[iind][jind])
            #print(Y[iind][jind])
            ax[iind, jind].plot(X[iind][jind],Y[iind][jind])
    if show:
        plt.show()
    return fig, ax

X =  [ 
                    np.array([[1,2,3], [4,5,6], [7,8,9]])
                ,
                
                    np.array([[11,12,13], [14,15,16], [17,18,19]])
                ,
                
np.array([[21,22,23], [24,25,26], [27,28,29]])
                
              ]
plot_grid(X,X,True,True)