import os 
import sys

sys.path.insert(0, '.') # make runable from src/

# external libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# For path referencing
from config.definitions import ROOT_DIR

# Python's built in libs
from collections import Counter

# Global constants
features_m = {
    'RI': 'refractive_index',
    'Na': 'sodium',
    'Mg': 'magnesium',
    'Al': 'aluminium',
    'Si': 'silicone',
    'K': 'potassium',
    'Ca': 'calcium',
    'Ba': 'barium',
    'Fe': 'iron'
}
features_names = ['refractive_index', 'sodium', 'magnesium', 'aluminium', 'silicone', 'potassium', 'calcium', 'barium', 'iron']
classes_m = {
    1: 'window_from_building_(float_processed)',
    2: 'window_from_building_(non_float_processed)',
    3: 'window_from_vehicle',
    5: 'container',
    6: 'tableware',
    7: 'headlamp'
}

def run_preprocessing():

    # Save the info about the process into a specified file
    old = sys.stdout
    out_path = os.path.join(ROOT_DIR, 'data', 'metadata', 'inspect_clean_transform_info.txt')
    sys.stdout = open(out_path, 'w')

    # Load data
    train = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'raw', 'df_train.csv'), delimiter=',', header=0)
    test = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'raw', 'df_test.csv'), delimiter=',', header=0)

    # Initial inspection
    print('-- Initial inspection ', end='-'*50 + '\n')
    print('Training data')
    print(train.head())
    print(end='\n\n')
    print('Test data')
    print(test.head())
    print(end='\n\n')
    print(f'Training data shape: {train.shape} | Test data shape: {test.shape}')
    print(f'There is in total {len(np.unique(train["type"]))} classes labeled as: {np.unique(train["type"])}')
    print(end='\n\n')


    # Split the data
    x_train, x_val, y_train, y_val = train_test_split(train.iloc[:, :-1],
                                                    train.iloc[:, -1],
                                                    test_size=0.33,
                                                    random_state=42)
    
    x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    # Define transformations method
    scaler = StandardScaler(with_mean=True, with_std=True) # Mean zero, unit variance
    pca = PCA(random_state=42)

    # Transform
    print('-- Label distribution ', end='-'*50 + '\n')
    print('\nMap from key to actual name:')
    print('-'*40)
    for k, v in classes_m.items():
        print(f'{k} --> {v}')
    print('-'*40)
    data = [[x_train, y_train, 'train'], [x_val, y_val, 'val'], [x_test, y_test, 'test']]
    expl_var = dict()
    for t in data:

        # Load and transform
        X, y, path = t
        X_scaled = scaler.fit_transform(X)
        X_pca = pca.fit_transform(X_scaled)
        expl_var[path] = pca.explained_variance_ratio_
        
        # Save
        X.to_csv(os.path.join(ROOT_DIR, 'data', 'transformed', path, 'X_org.csv'), index=False)
        pd.DataFrame(X_scaled).to_csv(os.path.join(ROOT_DIR, 'data', 'transformed', path, 'X_scaled.csv'), index=False, header=False)
        pd.DataFrame(X_pca).to_csv(os.path.join(ROOT_DIR, 'data', 'transformed', path, 'X_pca.csv'), index=False, header=False)
        y.to_csv(os.path.join(ROOT_DIR, 'data', 'transformed', path, 'y.csv'), index=False, header=False)
        
        # Show info about the process
        print('\n' + f"{path.title()} dataset sorted according to perc\n" + '-'*40)
        c = Counter(y)
        k_cn = [(k, cn,) for k, cn in c.items()]
        k_cn_sort = sorted(k_cn, key=lambda x: x[1], reverse=True)
        for t2 in k_cn_sort:
            k, cn = t2
            print(f'Key: {k} | Percent: {round(cn/y.shape[0]*100, 2)} %')
        print('-'*40 + '\n')

    # Info about transformation
    print('-- Transformation info ', end='-'*50 + '\n')
    print('Finished succesfully data transformation using standard scaling and pca.')
    print(f"Percentage of explained variance by first two components: {round(expl_var['train'][:2].sum()*100, 2)} %")
    print(end='\n\n')


    # End tracking the process
    sys.stdout.close()
    sys.stdout = old
