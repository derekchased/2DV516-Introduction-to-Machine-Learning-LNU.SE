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
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
"""

import assignment2_logistic_regression_functions as lorf
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
"""


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
    

# 1- Get X and y from insurance.csv
np_data, headers = load_data()

# 2a - Use simple label encoder instead of one hot encoder...
X, y = preprocess_Label_Encoder(np_data)

# 2b - Transform categorical data (some are binary, some are one hot encoded)
#X, y, headers = preprocess_One_Hot_Encoder(np_data, headers)


# 3- Normalize data using Standard Scalar (which uses the same method we used)
def plot_raw_data():
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    return Xn

# 4 Plot data 
def plot_raw_data():
    for ind in range(X.shape[1]):
        plot_feature(X[:,ind].reshape(-1,1),y.reshape(-1,1),'b',headers[ind])


# 5 - Linear Regression with basic split test
test_size = 0.2
random_state = 0
alpha = 1

def regression_tts(X,y,test_size,random_state,alpha,cclass,label):    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    #print("test_size ",test_size)
    try:
        regressor = cclass(alpha=alpha)  
    except:
        regressor = cclass()  
    regressor.fit(X_train, y_train) #training the algorithm
    #To retrieve the intercept:
    #print(regressor.intercept_)
    #For retrieving the slope:
    #print(regressor.coef_)
    
    y_pred = regressor.predict(X_test)
    mean_mse = np.mean(metrics.mean_squared_error(y_test, y_pred))
    
    print("=== TTS "+label+" ===")
    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', mean_mse)  
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("======")
    
    #for ind in range(X.shape[1]):
    #    plot_feature_pred(X_test[:,ind].reshape(-1,1),y_test.reshape(-1,1),y_pred.reshape(-1,1),'b',label + " " +headers[ind])
    

# Train Test Split
regression_tts(X, y,test_size,random_state,alpha,LinearRegression,"Standard Linear Regression")
regression_tts(X, y,test_size,random_state,alpha,Lasso,"Lasso Regression")
regression_tts(X, y,test_size,random_state,alpha,Ridge,"Ridge Regression")
regression_tts(X, y,test_size,random_state,alpha,ElasticNet,"ElasticNet Regression")



def regression_cvs(X,y,alpha,cclass,label,cv=5,
                   scoring="neg_root_mean_squared_error"):    
    #X_train, X_test, y_train, y_test = train_test_split(
     #   X, y, test_size=test_size, random_state=random_state)
    
    try:
        regressor = cclass(alpha=alpha)  
    except:
        regressor = cclass()  
    
    MSE = cross_val_score(regressor,X,y,scoring=scoring, cv = cv)
    
    y_pred = cross_val_predict(regressor,X,y,cv = cv)
    
    mean_mse = np.mean(MSE)
    
    print("=== CVS "+label+" ===")
    print('Mean MSE:', mean_mse)  
    print("======")
    
    plot_feature(y,y_pred,'b',label + " Predicted, MSE=",mean_mse)

def regression_grid_search_cvs(X,y,cclass,params,alphas,label,
                               cv=5,scoring="neg_root_mean_squared_error"):    
    #X_train, X_test, y_train, y_test = train_test_split(
     #   X, y, test_size=test_size, random_state=random_state)
    
    regressor = cclass()
    #lasso = Lasso(random_state=0, max_iter=10000)
    
    global gscv
    gscv = GridSearchCV(regressor, params, scoring=scoring, cv = cv)
    
    gscv.fit(X,y)
    
    print("=== GS "+label+", "+str(cv)+" Folds ===")
    print('Optimal Alpha: ', gscv.best_params_)
    print('Optimal MSE: ', abs(gscv.best_score_))
    #print("best_estimator_",gscv.best_estimator_)
    #print("best_index_",gscv.best_index_)
    #print(gscv.cv_results_)
    
    #scores = gscv.cv_results_['mean_test_score']
    #print('Scores: ', scores)
    #print(scores.shape)
    #scores_std = gscv.cv_results_['std_test_score']
    #print('scores_std: ', scores_std)
    #print(scores_std.shape)    
    """plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, scores)
    
    # plot error lines showing +/- std. errors of the scores
    std_error = scores_std / np.sqrt(cv)
    
    plt.semilogx(alphas, scores + std_error, 'b--')
    plt.semilogx(alphas, scores - std_error, 'b--')
    
    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
    
    plt.ylabel('CV score +/- std error')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([alphas[0], alphas[-1]])
    
    plt.show()"""
    
    
    
    

# Cross Val Score
regression_cvs(X, y, alpha, LinearRegression,"Standard Linear Regression")
alphas = np.linspace(.0001,100,100)
params = {"alpha":alphas}
#alphas = np.logspace(-4, -0.5, 30)
#alphas = np.array([1e-10, 1e-6,1e-8, 1e-4, 1e-3, 1e-2,1,5,10,20,50,100,1000])
#print(alphas)
regression_grid_search_cvs(X, y,Lasso,params,alphas,"Lasso Regression",)
regression_grid_search_cvs(X, y,Ridge,params,alphas,"Ridge Regression")
params = {"alpha":alphas,"l1_ratio":np.linspace(.01,1,10)}
regression_grid_search_cvs(X, y,ElasticNet,params,alphas,"ElasticNet Regression")




"""
def lasso_regression(X, y,test_size,random_state,alpha):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_state)

    reg = Lasso(alpha)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    print("\n\n=== 2. Lasso Regression ===")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("\n======")
    pass

#lasso_regression(X, y,test_size,random_state,alpha)

def ridge_regression(X, y,test_size,random_state,alpha):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_state)

    reg = Ridge(alpha)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    print("\n\n=== 3. Ridge Regression ===")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("\n======")
    pass

#ridge_regression(X, y,test_size,random_state,alpha)

def elastic_regression(X, y,test_size,random_state,alpha):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_state)

    reg = ElasticNet(alpha)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    print("\n\n=== 4. ElasticNet Regression ===")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("\n======")
    pass

#elastic_regression(X, y,test_size,random_state,alpha)
"""






def exercise7():
    print ("\nExercise 7")
    print("the goal is to predict insurance charges given certain",
          "traits of the policyholders")
    print("find a good linear regression model (you should determine",
          "on your own. what can be considered a relevant way to measure",
          "which model is best among your candidates")
    print("\nA)\n- standard linear regression\n- lasso regression\n- ridge ",
          "regression\n- elastic net regression")
    print("\nB)\nExtensions and further work can be for instance optimizing:",
          "\n- for the regularization hyperparameter λ (this is called alpha",
          "in sklearn),\n- adding transformed features (e.g. polynomial",
          "features),\n- combining transformed features and regularization.")
    print("\n-> Note that several of the variables in the dataset are",
          "categorical variables, and you’ll need to figure out a way to ",
          "treat these variables properly")


def exercise7_1():
    print ("\nExercise 7.1")

#X, y = load_data()




#exercise7()
#exercise7_1()



