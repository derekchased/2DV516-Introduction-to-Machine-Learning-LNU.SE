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

    # Step 1 - Load Data
    Csv_data = np.loadtxt("./A2_datasets_2020/housing_price_index.csv",delimiter=',') # load csv
    
    X = Csv_data[:,0:1]
    y = Csv_data[:,1]
    years = np.array(X)+1975 # for label on plot
    
    # Step 2 - Plot the data
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Ex 2.1, Housing price index Småland', fontsize=14)
    fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
    ax.set(xlabel="Year",ylabel="Price Index")
    ax.plot(years,y,c="r")
    
    # Step 3 and 4
    # 3- Find minimum cost by iterating over different degrees
    # 4- Plot the curve obtained for each degree we try
    
    # setup plot
    rows, cols = 2, 2
    fig, ax = plt.subplots(rows,cols,sharey=True,sharex=True)
    fig.suptitle('Ex 2.1, Polynomial fit', fontsize=14)
    fig.tight_layout(pad=1.0,rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, -.05, 'x', ha='center')
    fig.text(-0.01, 0.5, 'F(x)', va='center', rotation='vertical')
    
    # initalize vars for iterating over degrees
    Xe, mincost, ind, deg, j, k,b = None, None, None, 4,0,0,None
    for i in range(1,deg+1):
        # Compute vars
        Xe = alrf.extended_matrix_deg(X,i)
        b = alrf.normal_equation(Xe,y)
        cost = alrf.cost_function(Xe,b,y)
        pred = alrf.predict(Xe, b)
        
        # keep track of the minimum cost value
        try:
            if cost < mincost:
                mincost = cost
                ind = i
        except:
            mincost = cost
            ind = i
        
        # plot actual data (scatter points) against our curve (plot line)
        ax[j][k].scatter(years,y,s=1,c="r")
        ax[j][k].plot(years,pred)
        
        # xlabelposition="above"
        ax[j][k].set(xlabel="Degree "+str(i)+ ", MSE ="+str(round(cost,1)))
        k +=1
        if k==cols:
            j+=1
            k =0
        
    print("\nEx 2.2 - I believe that of these 4 degrees, that degree of 4",
          "gives the best fit to the data. Firstly, the MSE is lowest on",
          "this degree. While MSE of the training data is not terribyly",
          "reliable, it is one of the best indicators that we have for this",
          "exercise. Secondly, by looking at the plot against the data, ",
          "we can see that the line fits to the data points the best, it",
          "has the least amount of distance visually (which is proven",
          "by the minimal MSE. Finally, for the data presented, there are",
          "about 9 inflection points. As described in the lecture, the ",
          "number of inflection points is a good indicator of the degree",
          "of the polynomial- they should match up. Therefore, I would assume",
          "that a 9th degree polynomial fit would have a minimal MSE here,",
          "and fit the data points the closest. Indeed, I did test with 9",
          "degrees and the MSE was in fact at it's minimal with a 9 degree",
          "polynomial.")
    
    print("\nEx 2.3")
    
    print("\nA housing price index (HPI) measures the price changes of", 
          "residential housing as a percentage change from some specific",
          "start date (which has HPI of 100)- Wikipedia.")
    
    print("\nSo, we will start our analysis in 1980, which has a value of 100",
          "Jonas purchased his house in 2015 for 2.3 million, so in terms of",
          "1980, the price would be 0.404929577 million.")
    
    
    #index_2015, price_2015_million = 568, 2.3
    #ratio_2015 = 
    pred_2022_index = alrf.predict(alrf.extended_matrix_deg( np.array([[47]]),4), b)[0]
    
    # using {pred_2022_index} with some outside math done in excel
    pred_2022_price_in_millions = 3.235387324
    
    print("Now we can use our model to predict the housing index in 2022. Using",
          "the X value of 47 the predict housing index in 2022 is",
          str(round(pred_2022_index))," and the corresponding price is",
          str(round(pred_2022_price_in_millions,4)),"million sek")
    
    print("\nI do believe the answer is somewhat realistic, in the USA we would",
          "say 'it's in the ballpark'. However, the data has many inflection",
          "points and there is no way to tell exactly where the actual price will",
          "be in 2022- we don't know which way the data will inflect.")
    
        #print("Degree",i," - MSE",cost)
    #print("==>Degree",ind,"has a minimum cost of ", mincost)
    plt.show()