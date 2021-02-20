import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

# read dataset
iris_data = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# shuffle dataset
iris_data = iris_data.sample(frac = 1, random_state=42)

# format dataset
iris_data.reset_index(drop=True, inplace=True)
iris_data['class'] = iris_data['class'].astype('category')
iris_data.drop(['sepal_length', 'petal_length'], axis=1, inplace=True)
classes = iris_data['class'].unique()

# split into train and test sets
train_data = iris_data.iloc[:int(0.6*iris_data.index.size)]
test_data = iris_data.iloc[int(0.6*iris_data.index.size):]

# split into X and y
train_X = train_data[['sepal_width', 'petal_width']]
train_y = train_data['class']

test_X = test_data[['sepal_width', 'petal_width']]
test_y = test_data['class']

for criteria in ['entropy', 'gini']:
    clf_RF = RandomForestClassifier(3, criterion = criteria)
    clf_RF.fit(train_X, train_y)
    y_pred = clf_RF.predict(test_X)
    clf_RF.plot()
    y_reset = test_y.reset_index(drop=True)
    print('\n-------------------------------------')
    print('Criteria :', criteria)
    print('-------------------------------------')
    print('Accuracy: ', accuracy(y_pred, y_reset))
    for cls in classes:
        print('-------------------------------------')
        print('Class: ', cls)
        print('Precision: ', precision(y_pred, y_reset, cls))
        print('Recall: ', recall(y_pred, y_reset, cls))