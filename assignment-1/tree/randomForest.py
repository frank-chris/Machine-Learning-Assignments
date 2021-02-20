from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as tr

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        # initialising attributes
        self.trees_list = []
        self.features_chosen = []
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        # initialising estimators
        for i in range(self.n_estimators):
            self.trees_list.append(DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth))

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert(X.index.size == y.size)
        self.X = X
        self.y = y

        M = X.columns.size

        m = int(M**(1/2))

        for i, tree in enumerate(self.trees_list):
            # randomly picking m features 
            X_sample = X.sample(n=m,axis='columns',replace=True, random_state=i%2+1)
            self.features_chosen.append(list(X_sample.columns))
            tree.fit(X_sample, y)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = {}
        for i, tree in enumerate(self.trees_list):
            predictions[str(i)] = pd.Series(tree.predict(X[self.features_chosen[i]]))

        pred_df = pd.DataFrame(predictions)

        # returning mode of predictions of estimators as final prediction
        return pred_df.mode(axis=1)[0]

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        # plotting learnt trees
        for i, tree in enumerate(self.trees_list):
            fig1 = plt.figure(figsize=(5,5))
            fig1.suptitle('Decision Tree-'+str(i+1), fontsize=16)
            tr.plot_tree(tree, filled=True, feature_names=self.X.columns, class_names=self.y.unique())

        # plotting decision boundaries of individual estimators
        x_min, x_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:, 0].max() + 1
        y_min, y_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:, 1].max() + 1
        xx, yy = pd.Series(np.arange(x_min, x_max, 0.05)), pd.Series(np.arange(y_min, y_max, 0.05))
        xxyy = pd.DataFrame({'xx':xx, 'yy':yy})
        fig2, axarr = plt.subplots(1, self.n_estimators, sharex='col', sharey='row', figsize=(24, 5))

        for idx, clf, tt in zip([i for i in range(self.n_estimators)],
                                self.trees_list,
                                [str(i+1) for i in range(self.n_estimators)]):
            Z = pd.DataFrame()

            if self.features_chosen[idx][0] == self.X.columns[0]:
                for row in yy:
                    Z = Z.append(pd.Series(clf.predict(xxyy[['xx']])), ignore_index=True)
            else:
                for col in xx:
                    Z[col] = pd.Series(clf.predict(xxyy[['yy']]))
            
            to_replace = list(pd.unique(Z.values.ravel('K')))
            to_replace.sort()
            value = [i for i in range(len(to_replace))]
            
            Z = Z.replace(to_replace=to_replace, value=value)
            c = self.y.replace(to_replace=to_replace, value=value)

            axarr[idx].contourf(xx, yy, Z, alpha=0.5)
            axarr[idx].scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=c, s=20, edgecolor='k')
            axarr[idx].set_title(tt)

        for axis in axarr.flat:
            axis.set(xlabel='x1='+str(self.X.columns[0]), ylabel='x2='+str(self.X.columns[1]))

        for axis in axarr.flat:
            axis.label_outer()

        fig2.suptitle('Decision Boundaries of Estimators -RandomForestClassifier')
        
        # plotting overall decision boundary
        fig3, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 5))

        Z = pd.DataFrame()

        for col in xx:
            temp = pd.DataFrame()
            temp[self.X.columns[0]] = pd.Series([col for x in range(xxyy.index.size)])
            temp[self.X.columns[1]] = xxyy['yy']
            Z[col] = self.predict(temp)

        to_replace = list(pd.unique(Z.values.ravel('K')))
        to_replace.sort()
        value = [i for i in range(len(to_replace))]
        
        Z = Z.replace(to_replace=to_replace, value=value)
        c = self.y.replace(to_replace=to_replace, value=value)
        
        ax.contourf(xx, yy, Z, alpha=0.5)
        ax.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=c, s=25, edgecolor='k')
        ax.set_title('Overall Decision Boundary - RandomForestClassifier')
        ax.set(xlabel='x1='+str(self.X.columns[0]), ylabel='x2='+str(self.X.columns[1]))
        plt.show()


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        # intialising attributes
        self.trees_list = []
        self.features_chosen = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        # intialising estimators
        for i in range(self.n_estimators):
            self.trees_list.append(DecisionTreeRegressor(max_depth=self.max_depth))

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert(X.index.size == y.size)
        M = X.columns.size

        m = int(M**(1/2))

        for tree in self.trees_list:
            # picking m features randomly
            X_sample = X.sample(n=m,axis='columns',replace=True)
            self.features_chosen.append(list(X_sample.columns))
            tree.fit(X_sample, y)

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = {}
        for i, tree in enumerate(self.trees_list):
            predictions[str(i)] = pd.Series(tree.predict(X[self.features_chosen[i]]))

        pred_df = pd.DataFrame(predictions)
        # returning mode of predictions as final prediction
        return pred_df.mean(axis=1)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
