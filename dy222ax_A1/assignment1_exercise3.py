#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

@title: Assignment 1, Exercise 3


I have successfully downloaded the MNIST data using scikit learn's function, 
fetch_openml. I used the example found on their website as my starting point

https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html?highlight=mnist

I then used the knn functions that I developed in exercise 1 to easily 
classify the mnist data. I was able to process all 60000 train data points 
and 10000 test data points in approximately 4:40. The optimal K value is 3,
with the minimal amount of errors 259. 

I believe that this is a reasonable amount of time. This time is most likely
thanks to the optimization of the euclidean distance which I also optimized
during exercise 1. 

I first had a 1 loop implementation of the euclidean distance. However, I 
noticed that there were full vector implementations after investigating online.
After speaking with the TA's, they confirmed that the vector approach would be
even faster. For my first ML class in over ten years, this was very exciting!

... I have run the MNIST data here, and I have printed out the error rate for
K values from 1 to 31. This is essentially the elbow curve.

The optimal K value is therefore, the one that produces the least amount of 
errors. However, if processing or time is an issue, we can define a margin
and choose the lowest K value that produces the lowest errors within the margin.
For example, if k=11 is better than k=3, but k=3 runs in much less time and, is
only marginally better than k=3, then given our margin, we would choose k=3 as
the optimal solution.


Time to download data and compute MNIST for odd K 1 through 31

4:40 seconds 

K = 1
Num Errors: 266
Num Correct: 9734

K = 3
Num Errors: 259
Num Correct: 9741

K = 5
Num Errors: 279
Num Correct: 9721

K = 7
Num Errors: 292
Num Correct: 9708

K = 9
Num Errors: 305
Num Correct: 9695

K = 11
Num Errors: 316
Num Correct: 9684

K = 13
Num Errors: 332
Num Correct: 9668

K = 15
Num Errors: 337
Num Correct: 9663

K = 17
Num Errors: 354
Num Correct: 9646

K = 19
Num Errors: 371
Num Correct: 9629

K = 21
Num Errors: 375
Num Correct: 9625

K = 23
Num Errors: 390
Num Correct: 9610

K = 25
Num Errors: 392
Num Correct: 9608

K = 27
Num Errors: 395
Num Correct: 9605

K = 29
Num Errors: 410
Num Correct: 9590

K = 31
Num Errors: 420
Num Correct: 9580
"""

import numpy as np
import assignment1_knn_functions as knn_funcs
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


print(__doc__)

def mnist_knn():
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    # num data to use from the full set
    train_samples = 60000
    
    # num data to use for testing
    test_size = 10000
    
    # split downloaded data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=test_size)#000)
       
    # Calc distances between training and test sets
    sorted_distances_indeces = knn_funcs.train(X_train, X_test)
    
    # iterate over k values 
    for k in range(1,32,2):
        # get predictions of test set
        kZy = knn_funcs.get_knn(k, y_train, sorted_distances_indeces)
        
        # Get number of errors for benchmarking/elbow curve        
        num_errors = (np.equal( kZy, y_test) == False).sum()
        
        
        print("\nK =",k)
        print("Num Errors:",str(num_errors))
        print("Num Correct:",str(len(kZy)-num_errors))