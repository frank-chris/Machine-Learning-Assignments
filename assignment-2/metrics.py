import numpy as np
import pandas as pd

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)

    # y_bool is a pd.Series that contains True or False depending on
    # whether corresponding entries in y and y_hat match or not.
    y_bool = (y_hat == y)

    # y_bool.sum() gives the total no. of True values in y_bool
    return y_bool.sum()/y.size

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)

    # y_hat_cls is a pd.Series that contains True or False depending
    # on whether y_hat contains cls or not in the corresponding place
    y_hat_cls = (y_hat == cls)

    # y_cls is a pd.Series that contains True or False depending
    # on whether y contains cls or not in the corresponding place
    y_cls = (y == cls)

    # y_bool is a pd.Series that contains True or False depending
    # on whether both y_cls and y_hat_cls contain True in the 
    # corresponding place or not
    y_bool = (y_cls & y_hat_cls)

    if y_hat_cls.sum() == 0:
        return np.nan

    return y_bool.sum()/y_hat_cls.sum()

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)

    # y_hat_cls is a pd.Series that contains True or False depending
    # on whether y_hat contains cls or not in the corresponding place
    y_hat_cls = (y_hat == cls)

    # y_cls is a pd.Series that contains True or False depending
    # on whether y contains cls or not in the corresponding place
    y_cls = (y == cls)

    # y_bool is a pd.Series that contains True or False depending
    # on whether both y_cls and y_hat_cls contain True in the 
    # corresponding place or not
    y_bool = (y_cls & y_hat_cls)

    if y_cls.sum() == 0:
        return np.nan

    return y_bool.sum()/y_cls.sum()

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)

    error = y - y_hat

    squared_error = error**2

    mean_squared_error = squared_error.sum()/y.size

    root_mean_squared_error =  mean_squared_error**(1/2)

    return root_mean_squared_error


def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)

    absolute_error = (y_hat - y).abs()

    mean_absolute_error = absolute_error.sum()/y.size

    return mean_absolute_error