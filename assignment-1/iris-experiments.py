import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import accuracy, precision, recall

np.random.seed(42)

# reading dataset
iris_data = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# shuffling dataset
iris_data = iris_data.sample(frac = 1, random_state=42)

# formatting
iris_data.reset_index(drop=True, inplace=True)
iris_data['class'] = iris_data['class'].astype('category')
classes = iris_data['class'].unique()

# splitting into train and test sets
train_data = iris_data.iloc[:int(0.7*iris_data.index.size)]
test_data = iris_data.iloc[int(0.7*iris_data.index.size):]

# splitting into X and y 
train_X = train_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
train_y = train_data['class']

test_X = test_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
test_y = test_data['class']

# using decision tree on IRIS dataset
print('Using decision tree on IRIS dataset:\n')

for criteria in ['information_gain', 'gini_index']:
    print('\n-------------------------------------')
    print('Criteria :', criteria)
    print('-------------------------------------')
    tree = DecisionTree(criterion=criteria, max_depth=5)
    tree.fit(train_X, train_y)
    tree.plot()
    y_hat = tree.predict(test_X)
    y = test_y.reset_index(drop=True)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in classes:
        print('-------------------------------------')
        print('Class: ', cls)
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

# Nested cross-validation to find optimum depths
print('\nUsing nested cross-validation to find optimum depth:\n')

best_depths = {}

for i in range(5):
    print('\nFold:', i+1, '(test sets)\n')
    test_data = iris_data.iloc[int(i*0.2*iris_data.index.size):int((i+1)*0.2*iris_data.index.size)]
    train_data = pd.concat([iris_data.iloc[:int(i*0.2*iris_data.index.size)], iris_data.iloc[int((i+1)*0.2*iris_data.index.size):]])

    test_X = test_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    test_y = test_data['class']

    accuracies_for_diff_depth = []
    for depth in range(1, 8):
        print('\nDepth:', depth, '\n')
        accuracies_for_diff_val_set = []
        for j in range(5):
            print('\nFold:', j+1, '(validation sets)\n')
            val_data = train_data.iloc[int(j*0.2*train_data.index.size):int((j+1)*0.2*train_data.index.size)]
            final_train_data = pd.concat([train_data.iloc[:int(j*0.2*train_data.index.size)], train_data.iloc[int((j+1)*0.2*train_data.index.size):]])

            val_X = val_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
            val_y = val_data['class']

            final_train_X = final_train_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
            final_train_y = final_train_data['class']

            tree = DecisionTree(criterion='information_gain', max_depth=depth)
            tree.fit(final_train_X, final_train_y)
            y_hat = tree.predict(val_X)
            y = val_y.reset_index(drop=True)
            accuracies_for_diff_val_set.append(accuracy(y_hat, y))

        accuracies_for_diff_depth.append(sum(accuracies_for_diff_val_set)/5)
        print('Average accuracy across 5 folds (validation set):', accuracies_for_diff_depth[-1])
    
    best_depths['fold_'+str(i+1)] = accuracies_for_diff_depth.index(max(accuracies_for_diff_depth)) + 1

# printing best depths
for fold, depth in best_depths.items():
    print('\nBest depth for', fold, 'is', depth)

# finding metrics by using the best depth on each fold
for i in range(5):
    test_data = iris_data.iloc[int(i*0.2*iris_data.index.size):int((i+1)*0.2*iris_data.index.size)]
    train_data = pd.concat([iris_data.iloc[:int(i*0.2*iris_data.index.size)], iris_data.iloc[int((i+1)*0.2*iris_data.index.size):]])

    test_X = test_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    test_y = test_data['class']

    train_X = train_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    train_y = train_data['class']

    tree = DecisionTree(criterion='information_gain', max_depth=best_depths['fold_'+str(i+1)])
    tree.fit(train_X, train_y)
    y_hat = tree.predict(test_X)
    y = test_y.reset_index(drop=True)
    print('\nMetrics on fold_'+str(i+1)+' using best depth for the fold:\n')
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in classes:
        print('-------------------------------------')
        print('Class: ', cls)
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))
