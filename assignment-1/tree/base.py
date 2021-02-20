"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, information_gain_using_variance, gini_index, best_split

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth = None):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        # initialising attributes
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = {}

    def discrete_fit(self, examples, target, attributes, depth):
        '''
        fit() function for classification using ID3 algorithm
        Inputs:
        > examples : pd.DataFrame of samples
        > target : pd.Series of target attribute
        > attributes : list of attributes/features
        > depth : int denoting current depth of tree
        Returns:
        > the trained tree
        '''

        # base cases
        if target.unique().size == 1:
            return target.unique()[0]
        elif len(attributes) == 0:
            return target.value_counts().idxmax()
        elif self.max_depth != None and depth >= self.max_depth:
            return target.value_counts().idxmax()

        # finding best attribute to split on
        if self.criterion == 'information_gain':
            A = attributes[0]
            if examples[attributes[0]].dtype.name != 'category':
                split = best_split('entropy', target, examples[attributes[0]])
                attr_series = examples[attributes[0]].copy()
                attr_series[attr_series > split] = 'a'
                attr_series[attr_series != 'a'] = 'b'
                info_gain = information_gain(target, attr_series)
            else:
                info_gain = information_gain(target, examples[attributes[0]])
            for attr in attributes:
                if examples[attr].dtype.name != 'category':
                    split = best_split('entropy', target, examples[attr])
                    attr_series = examples[attr].copy()
                    attr_series[attr_series > split] = 'a'
                    attr_series[attr_series != 'a'] = 'b'
                    current_info_gain = information_gain(target, attr_series)
                else:
                    current_info_gain = information_gain(target, examples[attr])
                if current_info_gain > info_gain:
                    info_gain = current_info_gain
                    A = attr
        else:
            A = attributes[0]
            if examples[attributes[0]].dtype.name != 'category':
                split = best_split('entropy', target, examples[attributes[0]])
                attr_series = examples[attributes[0]].copy()
                attr_series[attr_series > split] = 'a'
                attr_series[attr_series != 'a'] = 'b'
                weighted_sum_gini_ind = 0
                for v in attr_series.unique():
                    weighted_sum_gini_ind += ((attr_series == v).sum()/attr_series.size)*gini_index(target[attr_series == v])
            else:
                weighted_sum_gini_ind = 0
                for v in examples[A].unique():
                    weighted_sum_gini_ind += ((examples[A] == v).sum()/examples[A].size)*gini_index(target[examples[A] == v])
            
            for attr in attributes:
                if examples[attr].dtype.name != 'category':
                    split = best_split('entropy', target, examples[attr])
                    attr_series = examples[attr].copy()
                    attr_series[attr_series > split] = 'a'
                    attr_series[attr_series != 'a'] = 'b'
                    current_weighted_sum_gini_ind = 0
                    for v in attr_series.unique():
                        current_weighted_sum_gini_ind += ((attr_series == v).sum()/attr_series.size)*gini_index(target[attr_series == v])
                else:
                    current_weighted_sum_gini_ind = 0
                    for v in examples[attr].unique():
                        current_weighted_sum_gini_ind += ((examples[attr] == v).sum()/examples[attr].size)*gini_index(target[examples[attr] == v])
                if current_weighted_sum_gini_ind < weighted_sum_gini_ind:
                    weighted_sum_gini_ind = current_weighted_sum_gini_ind
                    A = attr

        tree = {A:{}}

        # removing the attribute from list and recursing for discrete input
        # and recursing without removing attribute from list for real input
        if examples[A].dtype.name == 'category':
            attributes_a = attributes.copy()
            attributes_a.remove(A)
            for v in examples[A].unique():
                examples_v = examples[examples[A] == v]
                target_v = target[examples[A] == v]
                tree[A][v] = self.discrete_fit(examples_v, target_v, attributes_a, depth + 1)
        else:
            split = best_split('entropy', target, examples[A])
            examples_v = examples[examples[A] < split]
            target_v = target[examples[A] < split]
            if target_v.size == 0:
                tree[A]['<'+str(split)] = target.value_counts().idxmax()
            else:
                tree[A]['<'+str(split)] = self.discrete_fit(examples_v, target_v, attributes, depth + 1)
            examples_v = examples[examples[A] >= split]
            target_v = target[examples[A] >= split]
            if target_v.size == 0:
                tree[A]['>='+str(split)] = target.value_counts().idxmax()
            else:
                tree[A]['>='+str(split)] = self.discrete_fit(examples_v, target_v, attributes, depth + 1)

        return tree

    def real_fit(self, examples, target, attributes, depth):
        '''
        fit() function for regression using ID3 algorithm
        Inputs:
        > examples : pd.DataFrame of samples
        > target : pd.Series of target attribute
        > attributes : list of attributes/features
        > depth : int denoting current depth of tree
        Returns:
        > the trained tree
        '''
        # base cases
        if target.unique().size == 1:
            return target.unique()[0]
        elif len(attributes) == 0:
            return target.mean()
        elif self.max_depth != None and depth >= self.max_depth:
            return target.mean()

        # finding the best attribute to split on
        A = attributes[0]
        if examples[attributes[0]].dtype.name != 'category':
            split = best_split('variance', target, examples[attributes[0]])
            attr_series = examples[attributes[0]].copy()
            attr_series[attr_series > split] = 'a'
            attr_series[attr_series != 'a'] = 'b'
            info_gain = information_gain_using_variance(target, attr_series)
        else:
            info_gain = information_gain_using_variance(target, examples[attributes[0]])
        for attr in attributes:
            if examples[attr].dtype.name != 'category':
                split = best_split('variance', target, examples[attr])
                attr_series = examples[attr].copy()
                attr_series[attr_series > split] = 'a'
                attr_series[attr_series != 'a'] = 'b'
                current_info_gain = information_gain_using_variance(target, attr_series)
            else:
                current_info_gain = information_gain_using_variance(target, examples[attr])
            if current_info_gain > info_gain:
                info_gain = current_info_gain
                A = attr
        
        tree = {A:{}}

        # removing the attribute from list and recursing for discrete input
        # and recursing without removing attribute from list for real input
        if examples[A].dtype.name == 'category':
            attributes_a = attributes.copy()
            attributes_a.remove(A)
            for v in examples[A].unique():
                examples_v = examples[examples[A] == v]
                target_v = target[examples[A] == v]
                tree[A][v] = self.real_fit(examples_v, target_v, attributes_a, depth + 1)
        else:
            split = best_split('variance', target, examples[A])
            examples_v = examples[examples[A] < split]
            target_v = target[examples[A] < split]
            if target_v.size == 0:
                tree[A]['<'+str(split)] = target.mean()
            else:
                tree[A]['<'+str(split)] = self.real_fit(examples_v, target_v, attributes, depth + 1)
            examples_v = examples[examples[A] >= split]
            target_v = target[examples[A] >= split]
            if target_v.size == 0:
                tree[A]['>='+str(split)] = target.mean()
            else:
                tree[A]['>='+str(split)] = self.real_fit(examples_v, target_v, attributes, depth + 1)

        return tree
    
    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert(X.index.size == y.size)

        self.X = X
        self.y = y

        # calling the required fit function depending on regression/classification
        if y.dtype.name == 'category':
            self.tree = self.discrete_fit(X, y, list(X.columns), 0)
        else:
            self.tree = self.real_fit(X, y, list(X.columns), 0)

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        assert(X.columns.size == self.X.columns.size)

        # traversing through the tree
        y_list = []
        for i in range(len(X.index)):
            temp = self.tree
            while True:
                if type(temp) != dict:
                    y_list.append(temp)
                    break
                else:
                    feature = list(temp.keys())[0]
                    if X[feature].dtype.name == 'category':
                        value = X[feature].iloc[i]
                        temp = temp[feature][value]
                    else:
                        value = X[feature].iloc[i]
                        split = float(list(temp[feature].keys())[0][1:])
                        if value < split:
                            temp = temp[feature][list(temp[feature].keys())[0]]
                        else:
                            temp = temp[feature][list(temp[feature].keys())[1]]

        y = pd.Series(y_list)
        return y

    def plot(self, d=None, indent=0):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if d is None:
            d = self.tree
        for key, value in d.items():
            if key in self.X.columns:
                print('    '*indent + str(key) + ':')
            else:
                print('    '*indent + str(key) + '?')
            if isinstance(value, dict):
                self.plot(value, indent+1)
            else:
                print('    '*(indent+1) + str(value))
