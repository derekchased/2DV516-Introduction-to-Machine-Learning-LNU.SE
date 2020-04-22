#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  13 06:36:37 2020

@author: derek


@Instructions:
    - Certain quantitive questions in Exercise 1, can simply be handled as a 
    print statement in the program, ie: 
        ... What is the expected benchmark result for a certain graphics card? 
    - More qualitative questions should be handled as a comment in the notebook 
    or in a separate text file, ie:
        ... Motivate your choice of model.
    - All such answers can be grouped into a single text-file.
    - The non-mandatory VG-exercise will require a separate report.
"""

import assignment2_exerciseA
import assignment2_exercise1
import assignment2_exercise2
import assignment2_exerciseB
import assignment2_exercise3
import assignment2_exercise4
import assignment2_exercise5

def exerciseA():
    print ("\n\n<-- Assignment 2, Exercise A -->\n")
    assignment2_exerciseA.exerciseA_1()
    assignment2_exerciseA.exerciseA_1_gradient()
    
def exercise1():
    print ("\n\n<-- Assignment 2, Exercise 1 -->\n")
    assignment2_exercise1.exercise1_1()

def exercise2():
    print ("\n\n<-- Assignment 2, Exercise 2 -->\n")
    assignment2_exercise2.exercise2_1()

def exerciseB():
    print ("\n\n<-- Assignment 2, Exercise B -->\n")
    assignment2_exerciseB.exerciseB_1()

def exercise3():
    print ("\n\n<-- Assignment 2, Exercise 3 -->\n")
    assignment2_exercise3.exercise3_1()

def exercise4():
    print ("\n\n<-- Assignment 2, Exercise 3 -->\n")
    assignment2_exercise4.exercise4_1()
    assignment2_exercise4.exercise4_2()
    assignment2_exercise4.exercise4_4()

def exercise5():
    print ("\n\n<-- Assignment 2, Exercise 3 -->\n")
    assignment2_exercise5.exercise5_1()
    

exerciseA()
exercise1()
exercise2()
exerciseB()
exercise3()
exercise4()
exercise5()