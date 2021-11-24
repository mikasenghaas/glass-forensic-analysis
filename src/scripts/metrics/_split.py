# script includes function to quantify quality of split of data for decision tree constructing
import numpy as np
from collections import Counter

def binary_gini(y):
    p = len(y[y==0]) / len(y)
    return 2 * p * ( 1 - p ) 

def gini(y):
    N = len(y)
    counter = Counter(y)

    ans = 0
    for val in counter.values():
        ans += val / N * ( 1 - val / N )
    return ans

def entropy(y):
    N = len(y)
    counter = Counter(y)

    ans = 0
    for val in counter.values():
        ans += val / N * np.log(val / N)
    return -ans

def mse_split(y):
    # compute average of y vec
    y_mean = np.mean(y)
    return np.mean(np.sum((y-y_mean)**2))
