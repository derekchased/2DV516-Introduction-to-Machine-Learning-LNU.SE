#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 06:36:37 2020

@author: derek
"""

import assignment1_exercise1
import assignment1_exercise2
import assignment1_exercise3
import assignment1_exercise4

def exercise1():
    print ("\n<-- Assignment 1, Exercise 1 -->\n")
    assignment1_exercise1.exercise1_1()
    assignment1_exercise1.exercise1_2()
    assignment1_exercise1.exercise1_3()

def exercise2():
    print ("\n<-- Assignment 1, Exercise 2 -->\n")
    assignment1_exercise2.exercise2_1()
    assignment1_exercise2.exercise2_2()
    assignment1_exercise2.exercise2_3_and_4()
    assignment1_exercise2.exercise2_5()
    
def exercise3():
    print ("\n<-- Assignment 1, Exercise 3 -->\n")
    print ("PLEASE RUN THIS FUNCTION MANUALLY ===> PROCESSING TIME IS 4.5 minutes")
    #assignment1_exercise3.mnist_knn()
    
def exercise4():
    print ("\n<-- Assignment 1, Exercise 4 -->\n")
    assignment1_exercise4.exercise4_1()
    assignment1_exercise4.exercise4_2()
    assignment1_exercise4.exercise4_3()

exercise1()
exercise2()
exercise3()
exercise4()