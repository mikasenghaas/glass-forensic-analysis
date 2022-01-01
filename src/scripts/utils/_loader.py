# script that contains functionality to project data fast and conveniently 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import os
from config.definitions import ROOT_DIR

def get_data(raw=True, scaled=True, pca=False):
    if raw and not pca:
        
        BASEPATH = os.path.join(ROOT_DIR, 'data', 'raw')

        #print(np.loadtxt('../../data/raw/df_test.csv', skiprows=1, delimiter=','))
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

if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(raw=False, scaled=False, pca=False)

    print(f'X_train: {X_train.shape}')
    print(f'X_val: {X_val.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'y_val {y_val.shape}')
    print(f'y_test {y_test.shape}')
