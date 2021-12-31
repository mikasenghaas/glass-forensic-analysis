import os
import sys

sys.path.insert(0, os.path.abspath(''))

# external libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# custom imports
from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions
from scripts.utils import get_data

np.random.seed(1)

def main():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=False)

    # ------ constructing model ------

    # define pipeline (scaling, pca, decision tree)
    scaler = StandardScaler()
    scaler.fit(X_train)
    pipe = Pipeline(steps=[('scaler', scaler),
                           ('pca', PCA()),
                           ('knn', KNeighborsClassifier())])

    # define hyper parameters to grid search
    params = {
            'pca__n_components': list(range(1, X_train.shape[1]+1)),
            'knn__n_neighbors': list(range(1, 11)),
            'knn__weights': ['uniform', 'distance'] 
            }

    # define and train on grid
    grid = GridSearchCV(pipe, params)
    grid.fit(X_train, y_train)

    # report back best combination of hyperparameters
    print('Best N Neighbors:', grid.best_estimator_.get_params()['knn__n_neighbors'])
    print('Best Regulation Parameter C:', grid.best_estimator_.get_params()['knn__weights'])
    print('Best Number Of Components:', grid.best_estimator_.get_params()['pca__n_components']); print();

    # print summary of grid search
    # print(grid.cv_results_); print();

    # final model
    clf = grid.best_estimator_

    # evaluate performance
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
    print(f'Validation Score during Cross Validation: {grid.best_score_}')
    print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}'); print();

    print(confusion_matrix(y_test, test_preds))

if __name__ == '__main__':
    main()
