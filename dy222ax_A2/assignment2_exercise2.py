#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:49:50 2020

@author: Derek Yadgaroff, derek.chase84@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import assignment2_linear_regression_functions as alrf
import numpy as np

def exercise2_1():
    print ("\nExercise 2.1")
    pass

# Step 1 - Load Data
Csv_data = np.loadtxt("./A2_datasets_2020/housing_price_index.csv",delimiter=',') # load csv

X = Csv_data[:,0:1]
y = Csv_data[:,1]
years = np.array(X)+1975

# Step 2 - Plot the data
fig, ax = plt.subplots(1,1)
fig.suptitle('Ex 2, Småland', fontsize=14)
fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
ax.set(xlabel="Year",ylabel="Price Index")
ax.plot(years,y,c="r")

# Step 3 and 4
# 3- Find minimum cost by iterating over different degrees
# 4- Plot the curve obtained for each degree we try

rows = 2
cols = 2
fig, ax = plt.subplots(rows,cols,sharey=True,sharex=True)
fig.suptitle('Ex 2, Polynomial Fit', fontsize=14)
fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])

fig.text(0.5, -.05, 'x', ha='center')
fig.text(-0.01, 0.5, 'F(x)', va='center', rotation='vertical')

Xe, mincost, ind, deg, j, k = None, None, None, 4,0,0
for i in range(1,deg+1):
    # Compute 
    Xe = alrf.extended_matrix_deg(X,i)
    b = alrf.normal_equation(Xe,y)
    cost = alrf.cost_function(Xe,b,y)
    f_x = np.dot(Xe,b) #f(x) is the Xe matrix dot the betas
    
    # keep track of the minimum value
    try:
        if cost < mincost:
            mincost = cost
            ind = i
    except:
        mincost = cost
        ind = i
        
    ax[j][k].scatter(years,y,s=1,c="r")
    ax[j][k].plot(years,f_x,)
    
    # xlabelposition="above"
    ax[j][k].set(xlabel="Degree "+str(i)+ ", MSE ="+str(round(cost,1)))
    k +=1
    if k==cols:
        j+=1
        k =0
    print("Degree",i," - MSE",cost)
print("\n==>Degree",ind,"has a minimum cost of ", mincost)
plt.show()
"""
# Step 3 - Plot data
fig, ax = plt.subplots(2,3)
fig.suptitle('Ex A.1, Girl Height in inches', fontsize=14)
fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
titles = ["CudaCores","BaseClock","BoostClock","MemorySpeed",
          "MemoryConfig","MemoryBandwidth","BenchmarkSpeed"]
# iterate over columns of Xn by using the Transpose of Xn
i, j = 0,0
for ind, xi in enumerate(Xn.T):
    ax[i][j].scatter(xi,y)
    ax[i][j].set_title(titles[ind])
    j +=1
    if j==3: i,j = 1,0
plt.show()
"""




"""
# Step 2 - Normalize Data
Xn = alrf.feature_normalization(X)

# Step 3 - Plot data
fig, ax = plt.subplots(2,3)
fig.suptitle('Ex A.1, Girl Height in inches', fontsize=14)
fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
titles = ["CudaCores","BaseClock","BoostClock","MemorySpeed",
          "MemoryConfig","MemoryBandwidth","BenchmarkSpeed"]
# iterate over columns of Xn by using the Transpose of Xn
i, j = 0,0
for ind, xi in enumerate(Xn.T):
    ax[i][j].scatter(xi,y)
    ax[i][j].set_title(titles[ind])
    j +=1
    if j==3: i,j = 1,0
plt.show()


# Step 3a) - Compute β using the normal equation β = (XeT Xe)−1XeT y 
# where Xe is the extended nor- malized matrix 
# [1, X1, . . . , X6]. What is the predicted benchmark result for a 
# graphic card with the following (non-normalized) feature values?
# 2432, 1607, 1683, 8, 8, 256 The actual benchmark result is 114.

# Normal Equation
Xe = alrf.extended_matrix(X)
b = alrf.normal_equation(Xe,y)

# Predict
arr_pred = [[2432, 1607,1683,8, 8, 256]]
pred = alrf.extended_matrix(np.array(arr_pred))
y_pred = alrf.predict(pred,b)[0]
print("Predicted benchmark result", y_pred,"\nActual benchmark result 114",)

# Step 4 - What is the cost J(β) when using the β computed by 
# the normal equation above?
cost = alrf.cost_function(Xe,b,y)
print("The cost J(β) using the normal equation is",cost)


# Step 5 - Gradient Descent
# a)Find (and print) hyperparameters (α, N ) such that you get within 1% of 
# the final cost for the normal equation.

### Gradient
# Step 1 - Feature Normalize X
#X_fnormalized_grad = alrf.feature_normalization(Heights_X_parent)
#Xn

# Step 2 - Extend the normalized matrix
#Xe_grad = alrf.extended_matrix(X_fnormalized_grad)
Xe_n = alrf.extended_matrix(Xn)

# Step 3 - Set an initial beta. Fill it with zeros. Use size of num of vectors/columns
beta_grad_start = np.zeros(np.ma.size(Xe_n,1))

# Step 4 - Get the gradient descent array
alpha, n = .01, 10000
beta_grad_iterations = alrf.gradient_descent(Xe_n, y, beta_grad_start,alpha,n)

# Step 5 - Take the most recent/last beta value
beta_grad_final = beta_grad_iterations[-1]

# Step 6 - Calculate cost function for each beta
J_gradient = []
for i,j in enumerate(beta_grad_iterations):
    J_grad = alrf.cost_function(Xe_n,beta_grad_iterations[i],y)
    J_gradient.append(J_grad)
    
fig, ax1 = plt.subplots()
fig.suptitle('Ex A.1 Gradient Descent, alpha = '+str(alpha), fontsize=14)
ax1.set(xlabel="Number of iterations = "+str(len(beta_grad_iterations)),ylabel="Cost J, min = "+str(round(J_gradient[-1],3)))
ax1.plot(np.arange(0,len(beta_grad_iterations)),J_gradient)
plt.xlim(0, len(beta_grad_iterations))
plt.show()
grad_cost = J_gradient[-1] 
print("The J(β) using gradient descent is",str(grad_cost),"which is within",str(  100*abs(grad_cost-cost)/cost),"percent of normal cost. Less than 1%!")


# (b) What is the predicted benchmark result for the example 
# graphic card presented above?

# Step 1 - Create prediction array
#heights_to_predict = np.array([[65,70]])
#arr_pred

# Step 2 - Add heights that you want to predict to the other heights
#Heights_plus_pred = np.append(Heights_X_parent, heights_to_predict,0)
X_plus_pred = np.append(X, arr_pred,0)

# Step 3 - Normalize the new heights matrix
#Normalized_heights_plus_pred = alrf.feature_normalization( Heights_plus_pred )
X_plus_pred_normalized = alrf.feature_normalization( X_plus_pred )

# Step 4 - Extract the normalized heights that you need to predict
#Normalized_pred2 = np.array(  [Normalized_heights_plus_pred[-1]]   )
X_plus_pred_normalized_extracted = np.array(  [X_plus_pred_normalized[-1]]   )

# Step 5 - Extend the predictions matrix
#pred_ex_grad = alrf.extended_matrix(Normalized_pred2)
pred_ex_grad = alrf.extended_matrix(X_plus_pred_normalized_extracted)

# Step 6 - Predict the y
#y_parents_grad = alrf.predict(pred_ex_grad,beta_grad_final)
y_grad = alrf.predict(pred_ex_grad,beta_grad_final)
print("The predicted benchmark using gradient descent is:", y_grad[0])"""