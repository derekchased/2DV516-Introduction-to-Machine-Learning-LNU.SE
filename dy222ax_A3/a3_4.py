#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:41:19 2020

@author: derek

@objective:
    The dataset consists of 10 categories of different clothes, 
    and your overall objective is to find a feed-forward neural 
    network which can distinguish images on the different sets 
    of clothes. The dataset contains 60,000 images for training 
    and 10,000 for testing just as the ordinary MNIST. The images 
    are 28 × 28 pixels.

@notes:
    
    1. show 16 random samples
- reshape the images
- show imshow

2. gridsearch, MLPClassifier sklearn.neuralnetworks.
- layer size, hidden layer size, num layers
- activation function, step function, hyperbolic, relu

3. get confusion
- look at the common misclassified
- look where they are being classified as
- C num of classes
- sklearn.
- check for min max 0 and 255

4. train on the training set, predict train on train, get 
error against true 
- test: train on training set, but preditct on test points, compare against test truths

"""



import numpy as np

def load_data():
    train_data = np.loadtxt('./data/fashion-mnist_train.csv',delimiter=',',skiprows=1)
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    
    test_data = np.loadtxt('./data/fashion-mnist_test.csv',delimiter=',',skiprows=1)
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    
    
    return X_train, y_train, X_test, y_test


"""def print_MSE(clf,X_train,y_train,X_test,y_test):
    # Fit and then predict train
    clf.fit(X_train,y_train)
    preds = clf.predict(X_train)

    # Calc train MSE
    diff = preds - y_train
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("Train MSE",mse)

    # Calc test MSE
    preds = clf.predict(X_test)
    diff = preds - y_test
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("Test MSE",mse)"""
    
# Ex 1
def ex_4_1():
    print("\nA3, Ex3.1")


print("hello!")


"""
# Load Data
X_train, y_train, X_test, y_test = load_data()

# Set Random State
random_state = 42
#n_jobs = -1 # use parallel processing, auto choose num cores

print(X_train[0].shape)
print(X_train[0])

plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

    

ex_4_1()
#ex_3_2()
#ex_3_3()
"""
