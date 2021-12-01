import numpy as np
from ._autograd import Var

def convert_to_var(arr):
    shape = arr.shape

    arr = list(arr.flatten())
    for i in range(len(arr)):
        arr[i] = Var(float(arr[i]))

    return np.array(arr).reshape(shape)

def softmax(y_hot):
    return (y_hot.T / np.sum(y_hot, axis=1)).T
