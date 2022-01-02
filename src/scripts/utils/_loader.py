import numpy as np 
from sklearn.preprocessing import StandardScaler
import os
from config.definitions import ROOT_DIR

def get_data(raw=True, scaled=True, pca=False):

    """
    Loads transformed dara according to the given specificatiion.

    Parameters
    ----------
    raw : bool
        Do you want to return the raw data.
    
    scaled : bool
        Do you want to return data which were standard scaled - zero mean, unit variance.
    
    pca : bool
        Do you want to return data which were transformed using pca.
    
    Raises
    ------
    AssertionError
        If the specification of data to be returned does not match the available options.
    """

    if raw and not pca:
        
        BASEPATH = os.path.join(ROOT_DIR, 'data', 'raw')

        train = np.loadtxt(f'{BASEPATH}/df_train.csv', skiprows=1, delimiter=',')
        test = np.loadtxt(f'{BASEPATH}/df_test.csv', skiprows=1, delimiter=',')

        X_train, y_train = train[:, :-1], train[:, -1].astype(int)
        X_test, y_test = test[:, :-1], test[:, -1].astype(int)

        if scaled:
            train_n = len(X_train)
            test_n = len(X_test)

            X = np.vstack((X_train, X_test))

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            X_train, X_test = X[:train_n, :], X[train_n:, :]

        return X_train, X_test, y_train, y_test

    elif not raw:
        
        BASEPATH = os.path.join(ROOT_DIR, 'data', 'transformed')

        if scaled and not pca:
            X_train = np.loadtxt(f'{BASEPATH}/train/X_scaled.csv', delimiter=',') 
            X_val = np.loadtxt(f'{BASEPATH}/val/X_scaled.csv', delimiter=',') 
            X_test = np.loadtxt(f'{BASEPATH}/test/X_scaled.csv', delimiter=',') 
        elif pca and not scaled:
            X_train = np.loadtxt(f'{BASEPATH}/train/X_pca.csv', delimiter=',') 
            X_val = np.loadtxt(f'{BASEPATH}/val/X_pca.csv', delimiter=',') 
            X_test = np.loadtxt(f'{BASEPATH}/test/X_pca.csv', delimiter=',') 
        elif not scaled and not pca: 
            X_train = np.loadtxt(f'{BASEPATH}/train/X_org.csv', skiprows=1, delimiter=',') 
            X_val = np.loadtxt(f'{BASEPATH}/val/X_org.csv', skiprows=1, delimiter=',') 
            X_test = np.loadtxt(f'{BASEPATH}/test/X_org.csv', skiprows=1, delimiter=',') 
        else:
            assert False, 'something went wrong with your specifications'

        y_train = np.loadtxt(f'{BASEPATH}/train/y.csv', delimiter=',', dtype=int)
        y_val = np.loadtxt(f'{BASEPATH}/val/y.csv', delimiter=',', dtype=int)
        y_test = np.loadtxt(f'{BASEPATH}/test/y.csv', delimiter=',', dtype=int)

        return X_train, X_val, X_test, y_train, y_val, y_test

    else:
        assert False, 'something went wrong with your specifications'

