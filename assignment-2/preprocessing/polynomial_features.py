''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree = degree
        self.include_bias = include_bias

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        n_dim = len(X.shape)

        assert(n_dim == 1 or n_dim == 2)

        if n_dim == 1:
            X = X.reshape((1, X.shape[0]))

        transformed = []

        X_T = X.T

        for d in range(1, self.degree+1):
            for feature in X_T:
                arr = np.ones(X.shape[0])
                for i in range(d):
                    arr = np.multiply(arr, feature)
                transformed.append(arr)
        
        transformed = np.array([np.ones(X.shape[0])]+transformed)

        if self.include_bias:
            transformed = np.transpose(transformed)
        else:
            transformed = np.transpose(transformed[1:])

        if n_dim == 1:
            return transformed[0]
        else:
            return transformed
    
        
        
        
        
        
        
        
        
    
                
                
