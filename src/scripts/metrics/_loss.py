# script that contains commonly used loss-function in machine learning
import numpy as np
#from ..models.neural_net._autograd import Var

def se(y, p):
    return np.sum((y - p)**2)

def mse(y, p):
    return np.sum((y - p)**2) / Var(len(y))

def mae(y, p):
    return 1 / len(y) * sum(np.abs(y-p))

def zero_one_loss(y, p):
    return np.sum(y != p)

def binary_cross_entropy(y, p):
    return - (1 / len(y)) * np.sum(y * np.log(p) + ((1+y) * np.log(1-p)))

def cross_entropy(y, p):
    #return - (1 / y.shape[0]) * np.sum(y * np.log(p))
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    return -np.sum(np.log(p) * y)
