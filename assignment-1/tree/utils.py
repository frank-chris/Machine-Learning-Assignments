
import numpy as np
import pandas as pd

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    # p will store the probabilities of each class
    p = {}
    for label in Y:
        if label in p:
            p[label] += 1/Y.size
        else: 
            p[label] = 1/Y.size
    
    # H is the entropy calculated
    H = sum([-p[x]*np.log2(p[x]) for x in p])

    return H

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    # p will store the probabilities of each class
    p = {}
    for label in Y:
        if label in p:
            p[label] += 1/Y.size
        else: 
            p[label] = 1/Y.size

    gini_ind = 1 - sum([p[x]*p[x] for x in p])

    return gini_ind

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    assert(Y.size == attr.size)

    df = pd.DataFrame({'attr': attr, 'Y': Y})

    values = df['attr'].unique()

    sum_weighted_entropies = 0

    for v in values:
        Y_v = (df[df['attr'] == v])['Y']
        sum_weighted_entropies += (Y_v.size/Y.size)*entropy(Y_v)

    return (entropy(Y) - sum_weighted_entropies)

def information_gain_using_variance(Y, attr):
    """
    Function to calculate the information gain using variance instead of entropy
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    assert(Y.size == attr.size)

    df = pd.DataFrame({'attr': attr, 'Y': Y})

    values = df['attr'].unique()

    sum_weighted_entropies = 0

    for v in values:
        Y_v = (df[df['attr'] == v])['Y']
        sum_weighted_entropies += (Y_v.size/Y.size)*Y_v.var()

    return (Y.var() - sum_weighted_entropies)

def best_split(metric, Y, attr):
    """
    Function to calculate the best split for continuous/real attribute
    
    Inputs:
    > metric: 'variance' or 'entropy' depending on if it's regression or classification
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute for which the best split should be calculated
    Outputs:
    > Return the best split as a float
    """
    assert(Y.size == attr.size)

    data = pd.DataFrame({'attr': attr, 'Y': Y})

    sorted_data = data.sort_values(by = 'attr') 

    if metric == 'variance':
        variances = []
        for i in range(1, sorted_data['attr'].size):
            variances.append(sorted_data['Y'][:i].var() + sorted_data['Y'][i:].var())
        variances_series = pd.Series(variances)
        variances_series = variances_series.fillna(np.inf)
        min_index = variances_series.idxmin()
        return (sorted_data['attr'].iloc[min_index] + sorted_data['attr'].iloc[min_index+1])/2
    else:
        entropies = []
        for i in range(1, sorted_data['attr'].size):
            entropies.append(entropy(sorted_data['Y'][:i]) + entropy(sorted_data['Y'][i:]))
        min_index = entropies.index(min(entropies))
        return (sorted_data['attr'].iloc[min_index] + sorted_data['attr'].iloc[min_index+1])/2