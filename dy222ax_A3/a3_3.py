import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import a3_funcs as as3f

def load_data():
    train_data = np.loadtxt('./data/fbtrain.csv',delimiter=',')
    X_train = train_data[:, 0:-1]
    y_train = train_data[:, -1]
    
    test_data = np.loadtxt('./data/fbtest.csv',delimiter=',')
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    
    return X_train, y_train, X_test, y_test


def print_MSE(clf,X_train,y_train,X_test,y_test):
    # Calc train MSE
    preds = clf.predict(X_train)
    diff = preds - y_train
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("Train MSE",mse)

    # Calc test MSE
    preds = clf.predict(X_test)
    diff = preds - y_test
    diffsq = diff**2
    mse = np.sum(diffsq)/len(preds)
    print("Test MSE",mse)
    
# Ex 1
def ex_3_1():
    print("\nA3, Ex3.1")
    
    # Load Data
    X_train, y_train, X_test, y_test = load_data()
    
    # Set Random State
    random_state = 42
    #n_jobs = -1 # use parallel processing, auto choose num cores
    
    # Build a baseline DecisionTreeRegressor for comparison
    
    base_tree = DecisionTreeRegressor(random_state = random_state).fit(X_train, y_train)
    print("\nDecisionTreeRegressor no max depth set, Depth",base_tree.get_depth())
    print_MSE(base_tree, X_train, y_train, X_test, y_test)
    
    # Params for Grid Search
    dtrparams = {"max_depth":[1,2,3,4,5,6,7,8,9],"random_state":[random_state]}
                 
    # Cross validate and finetune hyperparameters
    gscv = as3f.grid_search_SVC(X_train, y_train, 
                                DecisionTreeRegressor, 5, dtrparams,print_score=False)

    # Get regressor from GridsearchCV
    clf = gscv.best_estimator_
    
    print("\nDecisionTreeRegressor Max Depth tuned, Depth", clf.get_depth())
    
    # Calc MSE
    print_MSE(clf,X_train,y_train,X_test,y_test)
    
    print("\nCan this be improved by not allowing the tree to grow too deep in the training process?")
    print("-> Yes, using gridsearchcv, we found that a max depth of 5 produced a better MSE result than not having a max depth. Without a max depth, we see the model created a depth of 42 with a worse MSE.")

# Ex 2
def ex_3_2():
    print("\nA3, Ex3.2")
    
    # Load Data
    X_train, y_train, X_test, y_test = load_data()
    
    # Set Random State
    random_state = 42
    #n_jobs = -1 # use parallel processing, auto choose num cores
    
    # Build a baseline DecisionTreeRegressor for comparison
    print("\nRandomForestRegressor default")
    base_tree = RandomForestRegressor(random_state = random_state,n_jobs=-1).fit(X_train,y_train)
    print_MSE(base_tree, X_train, y_train, X_test, y_test)
    
    # Params for Grid Search
    dtrparams = {"max_depth":[1,2,3,4,5,6,7,8,9],"random_state":[random_state],"n_jobs":[-1]}
                 
    # Cross validate and finetune hyperparameters
    gscv = as3f.grid_search_SVC(X_train, y_train, 
                                RandomForestRegressor, 5, dtrparams,print_score=False)
    print("\nRandomForestRegressor tuned")
    
    # Get regressor from GridsearchCV
    clf = gscv.best_estimator_
    
    # Calc MSE
    print_MSE(clf,X_train,y_train,X_test,y_test)

# Ex 2
def ex_3_3():
    print("\nA3, Ex3.3")
    
    # Load Data
    X_train, y_train, X_test, y_test = load_data()
    
    

#ex_3_1()
ex_3_2()
#ex_3_3()

