import os 
import sys

sys.path.insert(0, '.') # make runable from src/

# external libraries
import numpy as np 
import pandas as pd
import matplotlib as plt

# custom imports
from scripts.utils import get_data 

X_train, X_test, y_train, y_test = get_data(raw=True, scaled=False)


def run_check_data():
    print(any_missing_values(X_train, X_test))
    print(sum_of_chemical_composition(X_train, X_test))

def _check_missing_values(*args):
    # Any missing values
    missing_values = 0
    for arg in args:
        missing_values += pd.DataFrame(arg).isnull().sum().sum()

    return False if missing_values == 0 else True

def _sum_of_chemical_composition(X_train, X_test):
    total = np.vstack((X_train , X_test))
    
    return total[:, 1:].sum(axis=1)

if __name__ == '__main__':
    run_check_data()
