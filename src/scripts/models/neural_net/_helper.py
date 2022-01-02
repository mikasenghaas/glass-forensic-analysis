import numpy as np
from ._autograd import Var

def convert_to_var(arr):
    """Converts instances of an array to Var instances

    Parameters
    ----------
    arr : any dimensional array

    Returns
    -------
    The original shaped array with Var instances
    """
    shape = arr.shape

    arr = list(arr.flatten())
    for i in range(len(arr)):
        arr[i] = Var(float(arr[i]))

    return np.array(arr).reshape(shape)

def hot_encode(y, intcode):
    """Transform target y into a one hot encoded y

    Parameters
    ----------
    y : 1d array
    intcode : dict
        Keys are the code labels and values are the corresponding code ints

    Returns
    -------
    One hot encoded target y
    """
    k = len(intcode) 
    n = len(y)

    y_hot = np.empty((n, k)) 

    for i in range(n):
        for j in range(k):
            if j == intcode[y[i]]:
                y_hot[i, j] = 1
            else:
                y_hot[i, j] = 0

    return y_hot.astype(int)


def softmax(y):
    """Calculate softmax of a given array

    Parameters
    ----------
    y : 2d array
        n x p where n is a number of data points and p the 'probabilities' to 
        normalize

    """
    # y is a 2-dimensional matrix nxp
    return (np.exp(y.T) / np.sum(np.exp(y), axis=1)).T
