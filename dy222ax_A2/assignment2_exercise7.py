#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:26:11 2020

@author: derek
"""

import matplotlib.pyplot as plt
import numpy as np
import csv # for custom import and processing of csv data
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet # models
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import assignment2_matrix_functions as amf

def load_data():
    # open file and clean up data
    reader = csv.reader(open("./A2_datasets_2020/insurance.csv"), delimiter=";")
    
    # create list to store rows
    data = []
    
    # iterate over rows and ignore empty lines
    for row in reader:
        if(len(row) != 0):
            data.append(row)
    
    Xheaders = data[0][0:-1]
    # convert list to np array AND ignore the header row
    np_data = np.array(data[1:len(data)])
    
    return np_data, Xheaders

"""
Plot data for visualization
"""
def plot_feature(X,y,c,xlabel,ylabel="Charges"):
    fig, ax1 = plt.subplots()
    fig.suptitle(xlabel)
    ax1.scatter(X,y,s=1,c='b')
    ax1.plot([X.min(), X.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.show()

def plot_feature_pred(X,y_test,y_pred,c,xlabel,ylabel="Charges"):
    fig, ax1 = plt.subplots()
    fig.suptitle(xlabel)
    ax1.scatter(X,y_test,s=1,c='b')
    ax1.scatter(X,y_pred,s=1,c='r')
    plt.show()

"""
Encode categorical labels
- A) preprocess_Label_Encoder: assign ints to the regions (0,1, 2, 3)
- B) preprocess_One_Hot_Encoder: transform and give each region it's own column
     and use a binary value (0 0 1)
"""
def preprocess_Label_Encoder(np_data):
    # Create label object to change string representations to numbers
    le = LabelEncoder()
    
    # modify sex, column labels
    le.fit(["male","female"])
    sex = le.transform(np_data[:,1])
    
    # modify smoker column labels
    le.fit(["no","yes"])
    smoker = le.transform(np_data[:,4])
    
    # modify region, column labels
    le.fit(["northeast","southeast","southwest","northwest"])
    region = le.transform(np_data[:,5])
    
    # recombine all columns except the last AND convert to float
    X = np.c_[np_data[:,0], sex, np_data[:,2:4], 
              smoker, region, np_data[:,6:-1] ].astype(float)
    
    y = np_data[:,-1].astype(float)
    return X, y

def preprocess_One_Hot_Encoder(np_data,headers):
    # Create label object to change string representations to numbers
    le = LabelEncoder()
    
    # modify sex column labels to binary
    le.fit(["female","male"])
    sex = le.transform(np_data[:,1])
    
    # modify smoker column labels to binary
    le.fit(["no","yes"])
    smoker = le.transform(np_data[:,4])
    
    # modify region, column labels
    #le.fit(["northeast","southeast","southwest","northwest"])
    #region = le.transform(np_data[:,5])
    ohe = OneHotEncoder(sparse=False,drop='first')
    r = np_data[:,5].reshape(-1,1)
    region = ohe.fit_transform(r)    
    #region_cat_dropped = ohe.categories_[0][0]
    region_cols = ohe.categories_[0][1:4]
    headers[6:6] = region_cols.tolist()
    del( headers[5:6])
    
    # recombine all columns except the last AND convert to float
    X = np.c_[np_data[:,0], sex, np_data[:,2:4], 
              smoker, region, np_data[:,6:-1] ].astype(float)
    
    y = np_data[:,-1].astype(float)
    return X, y, headers

# 3- Normalize data using Standard Scalar (which uses the same method we used)
def plot_raw_data():
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    return Xn

# 4 Plot data 
def plot_raw_data():
    for ind in range(X.shape[1]):
        plot_feature(X[:,ind].reshape(-1,1),y.reshape(-1,1),'b',headers[ind])

# Regrssion using train_test_split
def regression_train_test_split(X,y,test_size,random_state,cclass,label):    
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    # Create Regression object
    regressor = cclass()  
        
    # Train the model
    regressor.fit(X_train, y_train) #training the algorithm
    
    # Get predictions
    y_pred = regressor.predict(X_test)
    
    # Get the MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    
    # Print out
    print("TTS "+label+", MSE="+str(mse))

# Regrssion using cross_val_score
def regression_cvs(X,y,cclass,cv,label,scoring="neg_root_mean_squared_error"):    
    
    # Create Regression object
    regressor = cclass()  
    
    # Obtain all MSE scores
    MSE = cross_val_score(regressor,X,y,scoring=scoring, cv = cv)
    
    # Get the average MSE 
    mean_mse = np.mean(MSE)
    
    print("CVS "+label+", K=" + str(cv) + ", MSE="+str(mean_mse))

# Regrssion using GridSearchCV
def regression_grid_search_cvs(X,y,cclass,cv,params,alphas,label,
                               scoring="neg_root_mean_squared_error"):    
    regressor = cclass()
    #lasso = Lasso(random_state=0, max_iter=10000)
    
    gscv = GridSearchCV(regressor, params, scoring=scoring, cv = cv)
    
    gscv.fit(X,y)
    
    print("GS_CVS "+label+", K=" + str(cv) + ", Optimal MSE="+str(abs(gscv.best_score_))+
          ", Optimal Alpha="+str(gscv.best_params_))

# 1- Get X and y from insurance.csv
np_data, headers = load_data()

# 2a - Use simple label encoder instead of one hot encoder...
X, y = preprocess_Label_Encoder(np_data)

# 2b - Transform categorical data (some are binary, some are one hot encoded)
#X, y, headers = preprocess_One_Hot_Encoder(np_data, headers)

# 5 - Init vars for standard linear regression
test_size = 0.2
random_state = 0
alphas = np.linspace(.001,100,100)
params = {"alpha":alphas,"max_iter":np.array([10000000])}

# Test Polynomials for all methods except Elastic (not enough memory/processor)
Xd = [X, amf.extended_matrix_deg(X,2,False)]
for Xr in Xd:
    
    # Regression using Train Test Split
    regression_train_test_split(Xr,y,test_size,random_state,
                                LinearRegression,"Standard Linear Regression")
    
    for cv in range(3,6):
        # Regression using Cross Val Score
        regression_cvs(Xr, y, LinearRegression,cv,"Standard Linear Regression")
        
        # Regression using Grid Search Cross Validation
        regression_grid_search_cvs(Xr, y,Lasso,cv,params,alphas,"Lasso Regression",)
        regression_grid_search_cvs(Xr, y,Ridge,cv,params,alphas,"Ridge Regression")
        

for cv in range(3,6):
    # For Elastic Net, add array of l1_ratio values to params
    params = {"alpha":alphas,"l1_ratio":np.linspace(.1,.99,100)}
    regression_grid_search_cvs(X, y,ElasticNet,cv,params,alphas,"ElasticNet Regression")
    



