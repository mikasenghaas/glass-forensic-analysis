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
    return -np.sum(np.log(p) * y)

if __name__ == '__main__':
    p = np.array([[0.25,0.25,0.25,0.25],
                  [0.01,0.01,0.01,0.96]])
    y = np.array([[0,0,0,1],
                  [0,0,0,1]])

    ans = 0.71355817782  #Correct answer
    x = cross_entropy(y, p)
    print(np.isclose(x,ans))
