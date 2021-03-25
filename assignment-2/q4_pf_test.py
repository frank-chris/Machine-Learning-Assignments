import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures



X = np.array([1,2])
print('\nInput 1-D array:')
print(X)
poly = PolynomialFeatures(2)
print('\nOutput 1-D array(degree = 2):')
print(poly.transform(X))

# a more complex example - 2D numpy array
X = np.arange(9).reshape((3,3))
print('\nInput 2-D array:')
print(X)
poly = PolynomialFeatures(2)
print('\nOutput 2-D array(degree = 2):')
print(poly.transform(X))
