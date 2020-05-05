import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import assignment_3_funcs as as3f
import matplotlib.pyplot as plt
import plt_functions as pltf
import assignment2_matrix_functions as a2mf
import assignment_3_funcs as as3f

def load_data():
    print("load_data")
    train_data = np.loadtxt('./data/fbtrain.csv',delimiter=',')
    X_train = train_data[:, 0:-1]
    y_train = train_data[:, -1]
    
    test_data = np.loadtxt('./data/fbtest.csv',delimiter=',')
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    
    return X_train, y_train, X_test, y_test


# Load Data
X_train, y_train, X_test, y_test = load_data()

# Normalize
#X_train = a2mf.feature_normalization(X_train)
#X_test = a2mf.feature_normalization(X_test)

# Set Random State
random_state = 42
n_jobs = -1

# Create Regressor and train
clf = DecisionTreeRegressor(random_state = random_state)

clf.fit(X_train,y_train)
preds = clf.predict(X_train)

diff = preds - y_train
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nDtree, train")
print("num correct",np.sum(preds==y_train))
print("num incorrect",np.sum(preds != y_train))
print("accuracy",np.sum(preds == y_train)/len(y_train))
print("mse",mse)


preds = clf.predict(X_test)
diff = preds - y_test
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nDtree, test")
print("num correct",np.sum(preds==y_test))
print("num incorrect",np.sum(preds != y_test))
print("accuracy",np.sum(preds == y_test)/len(y_test))
print("mse",mse)

clf = DecisionTreeRegressor(max_depth=3, random_state = random_state)

clf.fit(X_train,y_train)
preds = clf.predict(X_train)

diff = preds - y_train
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nDtree maxd=3, train")
print("num correct",np.sum(preds==y_train))
print("num incorrect",np.sum(preds != y_train))
print("accuracy",np.sum(preds == y_train)/len(y_train))
print("mse",mse)


preds = clf.predict(X_test)
diff = preds - y_test
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nDtree maxd=3, test")
print("num correct",np.sum(preds==y_test))
print("num incorrect",np.sum(preds != y_test))
print("accuracy",np.sum(preds == y_test)/len(y_test))
print("mse",mse)

# Create Regressor and train
clf = RandomForestRegressor(random_state = random_state,n_jobs=n_jobs)

clf.fit(X_train,y_train)
preds = clf.predict(X_train)

diff = preds - y_train
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest, train")
print("num correct",np.sum(preds==y_train))
print("num incorrect",np.sum(preds != y_train))
print("accuracy",np.sum(preds == y_train)/len(y_train))
print("mse",mse)


preds = clf.predict(X_test)
diff = preds - y_test
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest, test")
print("num correct",np.sum(preds==y_test))
print("num incorrect",np.sum(preds != y_test))
print("accuracy",np.sum(preds == y_test)/len(y_test))
print("mse",mse)

clf = RandomForestRegressor(max_depth=3, random_state = random_state,n_jobs=n_jobs)

clf.fit(X_train,y_train)
preds = clf.predict(X_train)

diff = preds - y_train
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest maxd=3, train")
print("num correct",np.sum(preds==y_train))
print("num incorrect",np.sum(preds != y_train))
print("accuracy",np.sum(preds == y_train)/len(y_train))
print("mse",mse)


preds = clf.predict(X_test)
diff = preds - y_test
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest maxd=3, test")
print("num correct",np.sum(preds==y_test))
print("num incorrect",np.sum(preds != y_test))
print("accuracy",np.sum(preds == y_test)/len(y_test))
print("mse",mse)



# Params for Grid Search
dctparams = {"criterion":["mse", "friedman_mse","mae"],
             "splitter":["best","random"],
             "max_depth":[3,4,5,6,7,8,9,10,11,12,13,14]}
             #"min_samples_split":[2,3,4,5,6,7,8,9,10,11,12,13,14],
             #"min_samples_leaf":[1,2,3,4,5,6,7,8,9,10,11,12,13,14]}
#gscv = as3f.grid_search_SVC(X_train, y_train, DecisionTreeRegressor, 5, dctparams)






print(y_train)
X_train, y_train, X_test, y_test



# Create Regressor and train
clf = RandomForestRegressor(random_state = random_state,n_jobs=n_jobs)

clf.fit(X_train,y_train)
preds = clf.predict(X_train)

diff = preds - y_train
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest, train")
print("num correct",np.sum(preds==y_train))
print("num incorrect",np.sum(preds != y_train))
print("accuracy",np.sum(preds == y_train)/len(y_train))
print("mse",mse)


preds = clf.predict(X_test)
diff = preds - y_test
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest, test")
print("num correct",np.sum(preds==y_test))
print("num incorrect",np.sum(preds != y_test))
print("accuracy",np.sum(preds == y_test)/len(y_test))
print("mse",mse)

clf = RandomForestRegressor(max_depth=3, random_state = random_state,n_jobs=n_jobs)

clf.fit(X_train,y_train)
preds = clf.predict(X_train)

diff = preds - y_train
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest maxd=3, train")
print("num correct",np.sum(preds==y_train))
print("num incorrect",np.sum(preds != y_train))
print("accuracy",np.sum(preds == y_train)/len(y_train))
print("mse",mse)


preds = clf.predict(X_test)
diff = preds - y_test
diffsq = diff**2
mse = np.sum(diffsq)/len(preds)
print("\nRForest maxd=3, test")
print("num correct",np.sum(preds==y_test))
print("num incorrect",np.sum(preds != y_test))
print("accuracy",np.sum(preds == y_test)/len(y_test))
print("mse",mse)


