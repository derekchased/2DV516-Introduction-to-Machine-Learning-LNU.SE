import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import assignment_3_funcs as as3f

"""
from sklearn.model_selection import cross_val_score
import assignment_3_funcs as as3f
import matplotlib.pyplot as plt
import plt_functions as pltf
import assignment2_matrix_functions as a2mf
"""

def load_data():
    train_data = np.loadtxt('./data/fbtrain.csv',delimiter=',')
    X_train = train_data[:, 0:-1]
    y_train = train_data[:, -1]
    
    test_data = np.loadtxt('./data/fbtest.csv',delimiter=',')
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    
    return X_train, y_train, X_test, y_test


# Ex 1
def ex_1():
    print("\nA3, Ex2.1")
    
    # Load Data
    X_train, y_train, X_test, y_test = load_data()
    
    # Set Random State
    random_state = 42
    #n_jobs = -1 # use parallel processing, auto choose num cores
    
    # Params for Grid Search
    dtrparams = {"max_depth":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                 "min_samples_split":[2,3,4,5,6,7,8,9,10,11,12,13,14],
                 "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                 "max_features":["auto","sqrt","log2"],
                 "random_state":[random_state]}
    
    
    gscv = as3f.grid_search_SVC(X_train, y_train, 
                                DecisionTreeRegressor, 5, dtrparams)
    
    
    # Create Regressor and train
    clf = DecisionTreeRegressor(random_state = random_state)
    print("\n\nDecisionTreeRegressor")
    
    # Fit and then predict train
    clf.fit(X_train,y_train)
    preds = clf.predict(X_train)
    
    # Calc train MSE
    diff = preds - y_train
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("Train MSE",mse)
    print("num correct",np.sum(preds==y_train))
    print("num incorrect",np.sum(preds != y_train))
    print("accuracy",np.sum(preds == y_train)/len(y_train))
    
    
    # Predict test and calc test MSE
    preds = clf.predict(X_test)
    diff = preds - y_test
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("\nTest MSE",mse)
    print("num correct",np.sum(preds==y_test))
    print("num incorrect",np.sum(preds != y_test))
    print("accuracy",np.sum(preds == y_test)/len(y_test))


# Ex 2
def ex_2():
    print("\nA3, Ex2.2")
    
    # Load Data
    X_train, y_train, X_test, y_test = load_data()
    
    # Set Random State
    random_state = 42
    n_jobs = -1 # use parallel processing, auto choose num cores
    
    # Create Regressor and train
    clf = RandomForestRegressor(random_state = random_state,n_jobs=n_jobs)
    print("\nRandomForestRegressor")
    
    clf.fit(X_train,y_train)
    preds = clf.predict(X_train)
    
    diff = preds - y_train
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("Train MSE",mse)
    print("num correct",np.sum(preds==y_train))
    print("num incorrect",np.sum(preds != y_train))
    print("accuracy",np.sum(preds == y_train)/len(y_train))
    
    
    preds = clf.predict(X_test)
    diff = preds - y_test
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("Test MSE",mse)
    print("num correct",np.sum(preds==y_test))
    print("num incorrect",np.sum(preds != y_test))
    print("accuracy",np.sum(preds == y_test)/len(y_test))


ex_1()
#ex_2()



