# external libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# custom imports
from scripts.plotting import plot_2d_decision_regions
from scripts.utils import get_data

np.random.seed(1)

def main():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=False)

    if False:
        engineer = PolynomialFeatures(degree = 2) 
        X_train = engineer.fit_transform(X_train)
        X_test = engineer.fit_transform(X_test)
        print(X_train.shape)

    # ------ constructing model ------

    # define pipeline (scaling, pca, decision tree)
    scaler = StandardScaler()
    scaler.fit(X_train) # use scaling parameters from training set in pipeline
    pipe = Pipeline(steps=[('scaler', scaler),
                           #('pca', PCA()),
                           ('random_forest', RandomForestClassifier(random_state=1))])

    # define hyper parameters to grid search
    params = {'random_forest__n_estimators': [20, 50, 100],
              'random_forest__max_depth': list(range(2, 15+1)),
              'random_forest__bootstrap': [False, True],
              'random_forest__criterion': ['entropy', 'gini'],
              #'pca__n_components': list(range(1, X_train.shape[1]-3)),
             }

    # define and train on grid
    grid = GridSearchCV(pipe, params, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # report back best combination of hyperparameters
    print('Best Number of Estimators', grid.best_estimator_.get_params()['random_forest__n_estimators'])
    print('Best Criterion', grid.best_estimator_.get_params()['random_forest__criterion'])
    print('Best Max Depth', grid.best_estimator_.get_params()['random_forest__max_depth'])
    print('Bootstrap Training Samples', grid.best_estimator_.get_params()['random_forest__bootstrap'])
    #print('Best Number Of Components:', grid.best_estimator_.get_params()['pca__n_components']); print();

    # final model
    clf = grid.best_estimator_
    """
    #Estimators: 50
    Criterion: entropy
    Max Depth: 
    Bootstrap: False
    #PCA Components: 
    """

    # evaluate performance
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
    print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}'); print();

    print(confusion_matrix(y_test, test_preds))


if __name__ == '__main__':
    main()
