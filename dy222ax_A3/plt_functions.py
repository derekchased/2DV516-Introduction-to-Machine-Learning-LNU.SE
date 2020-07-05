#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:42:20 2020

@author: derek
"""

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def get_meshgrid(X1, X2, h=.01):
    # Create meshgrid
    x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))  # Mesh Grid
    return xx, yy

def add_countour(ax, xx, yy, IPredictable, **params):

    # predict using the meshgrid's ravel
    # IPredictable should implement a predict function
    Z = IPredictable.predict(np.c_[xx.ravel(), yy.ravel()])

    # Map of predictions
    Z = Z.reshape(xx.shape)

    # Show the boundary
    ax.contour(xx, yy, Z, **params)

    return ax, Z


def add_countourf(ax, xx, yy, IPredictable, **params):
    # predict using the meshgrid's ravel
    # IPredictable should implement a predict function
    Z = IPredictable.predict(np.c_[xx.ravel(), yy.ravel()])

    # Map of predictions
    Z = Z.reshape(xx.shape)

    # Show the boundary
    # ax.contourf(xx, yy, Z, **params)

    # Create colors
    cmap_light = ListedColormap(
        ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#AB3217", "#407a51"])  # mesh plot
    # cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])  # colors

    # Show the boundary
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

    return ax, Z


def iterate_subplots(id, figsize, nrows, ncols):
    plt.figure(id, figsize=figsize)
    return [plt.subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]

    # axes = []
    # plt.figure(id, figsize=figsize)
    # for i in range(nrows * ncols):
    #     axes.append(plt.subplot(nrows, ncols, i + 1))
    # return axes
    
