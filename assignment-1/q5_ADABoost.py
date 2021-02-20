"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn.tree import DecisionTreeClassifier
# from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
Classifier_AB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features

print('\nUsing AdaBoostClassifier on IRIS dataset:\n')

# reading dataset
iris_data = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# shuffling dataset
iris_data = iris_data.sample(frac = 1, random_state=42)

# formatting dataset
iris_data.reset_index(drop=True, inplace=True)
iris_data['class'] = iris_data['class'].astype('category')
iris_data.drop(['sepal_length', 'petal_length'], axis=1, inplace=True)
iris_data.replace({'Iris-virginica':'virginica', 'Iris-versicolor':'not-virginica', 'Iris-setosa':'not-virginica'}, inplace=True)
classes = iris_data['class'].unique()

# splitting into train and test sets
train_data = iris_data.iloc[:int(0.6*iris_data.index.size)]
test_data = iris_data.iloc[int(0.6*iris_data.index.size):]

# splitting into X and y
train_X = train_data[['sepal_width', 'petal_width']]
train_y = train_data['class']

test_X = test_data[['sepal_width', 'petal_width']]
test_y = test_data['class']

# using AdaBoostClassifier on dataset
clf_AB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier, n_estimators=n_estimators )
clf_AB.fit(train_X, train_y)
y_pred = clf_AB.predict(test_X)
clf_AB.plot()
y_reset = test_y.reset_index(drop=True)

print('Accuracy: ', accuracy(y_pred, y_reset))
for cls in classes:
    print('-------------------------------------')
    print('Class: ', cls)
    print('Precision: ', precision(y_pred, y_reset, cls))
    print('Recall: ', recall(y_pred, y_reset, cls))

# using decision stump on dataset
print('\nUsing Decision Stump on IRIS dataset for comparison:\n')

clf_DS = DecisionTreeClassifier(criterion='entropy', max_depth=1)
clf_DS.fit(train_X, train_y)
y_pred_DS = clf_DS.predict(test_X)
y_reset = test_y.reset_index(drop=True)

print('Accuracy: ', accuracy(y_pred_DS, y_reset))
for cls in classes:
    print('-------------------------------------')
    print('Class: ', cls)
    print('Precision: ', precision(y_pred_DS, y_reset, cls))
    print('Recall: ', recall(y_pred_DS, y_reset, cls))