# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:29:29 2021

@author: mjvat
"""

import sys
import numpy as np
import pandas as pd

if __name__ == '__main__':
    input1 = str(sys.argv[1])
    output1 = str(sys.argv[2])
   
    data = pd.read_csv(input1)
    #data = pd.read_csv("E:/Transfer/Desktop/Schools/Columbia/Python Projects/Project 3 - Machine Learning/input1.csv")
    n = data.shape[0]
    X = np.column_stack((data.iloc[:, :-1], np.ones(shape=(n,))))
    y = data.iloc[:, -1]
   
    rows, cols = X.shape
    weights = np.zeros(cols)
    outputs = np.empty(shape=(0, cols), dtype=np.int)

    convergence = False
    while not convergence:
        convergence = True
        for i in range(n):
            xi, yi = X[i], y[i]
            preds = np.sign(np.sum(np.multiply(xi, weights)))
            
            if yi * preds <= 0:
                convergence = False
                for j in range(cols):
                    weights[j] += yi * xi[j]

        if convergence:
            break

        outputs = np.row_stack((outputs, weights))

    np.savetxt(fname=output1, X=outputs, fmt='%d', delimiter=",")