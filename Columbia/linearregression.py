# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:29:47 2021

@author: mjvat
"""

import os
import sys
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression 

def write_to_csv(results):
    with open('output2.csv','w') as output_file:
        output_file.writelines(results)
    output_file.close()

if __name__ == '__main__':    
    input2 = str(sys.argv[1])
    output2 = str(sys.argv[2])
   
    data = pd.read_csv(input2)
    #data = pd.read_csv('E:/Transfer/Desktop/Schools/Columbia/Python Projects/Project 3 - Machine Learning/input2.csv', header=None)
    
    # Features age, weight, height. Height is the 'label' column (in meters)
    features = data.iloc[:,:2]
    
    #features = features_copy
    labels = data.iloc[:,2]
    
    # Add the vector 1 (intercept) ahead of your data matrix
    vector1 = np.ones(shape=(len(features),1))
    features.insert(0,'vector1',vector1,True)
    
    # Find mean and SD of each feature
    mean = np.mean(features)
    mean = mean[1:,]
    
    sd = np.std(features)
    sd = sd[1:,]

    # Scale each feature (age and weight) by its population SD, which sets its mean to zero
    scaled_features = features
    scaled_features = scaled_features.drop(['vector1'], axis = 1)
    scaled_features = scale(scaled_features.iloc[:,:])

    # Run the GD using the following learning rates {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}
    alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 3.7]
    
    # Initialize beta's to zero
    beta = [0, 0, 0]
    beta = np.array(beta, dtype=np.float64)
    
    # For each a-value, run the aglorithm exactly 100 iterations. 
    iterations = [100, 100, 100, 100, 100, 100, 100, 100, 100, 373]
        # Compare the convergence rate when a is small vs. large
        # What is the ideal learning rate to obtain an accurate model?
    
    # Implement Gradient Descent to find the regression model.
    weights = np.zeros(3)
    results = []

    for alpha, iterations in zip(alpha, iterations):
        for i in range(iterations):
            preds = np.dot(features, weights)
            error = preds - labels
            gradient = (features.T * error).sum(axis=1)
            weights = weights - ((alpha/len(labels)) * gradient)
        weights = weights.tolist()
        results.append(','.join(map(str, [alpha, iterations, weights[-1], *weights[:-1]])) + '\n')    
    
    # Output data to output2.csv
    # output(alpha, number_of_iterations, b_0, b_age, b_weight)
    write_to_csv(results)
    
    