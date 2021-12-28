import numpy as np
from ._autograd import Var

def convert_to_var(arr):
    shape = arr.shape

    arr = list(arr.flatten())
    for i in range(len(arr)):
        arr[i] = Var(float(arr[i]))

    return np.array(arr).reshape(shape)

def hot_encode(y, intcode):
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
    # y is a 2-dimensional matrix nxp
    return (np.exp(y.T) / np.sum(np.exp(y), axis=1)).T
