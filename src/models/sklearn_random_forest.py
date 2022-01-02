import os
import sys

sys.path.insert(0, os.path.abspath('.'))

# external libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# custom imports
from scripts.plotting import plot_2d_decision_regions
from scripts.metrics import accuracy_score, confusion_matrix
from scripts.utils import get_data, generate_summary

np.random.seed(1)

SHOW = True

def run_sklearn_random_forest():
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
              'random_forest__max_depth': list(range(5, 10)) + [None],
              'random_forest__bootstrap': [False, True],
              'random_forest__criterion': ['entropy', 'gini'],
              #'pca__n_components': list(range(1, X_train.shape[1]-3)),
             }

    # define and train on grid
    grid = GridSearchCV(pipe, params, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # final model
    clf = grid.best_estimator_

    # report back best combination of hyperparameters
    best_n_estimators = grid.best_estimator_.get_params()['random_forest__n_estimators']
    best_criterion = grid.best_estimator_.get_params()['random_forest__criterion']
    best_max_depth = grid.best_estimator_.get_params()['random_forest__max_depth']
    best_bootstrap = grid.best_estimator_.get_params()['random_forest__bootstrap']


    if SHOW:
        print('-'*5 + ' Best Hyperparameters ' + '-'*5)
        print('Best N Estimators:', best_n_estimators)
        print('Best Criterion:', best_criterion)
        print('Best Max Depth:', best_max_depth)
        print('Best bootstrap:', best_bootstrap)


    # evaluate performance
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    train_acc = round(accuracy_score(y_train, train_preds), 2)*100
    val_acc = round(grid.best_score_, 2)*100
    test_acc = round(accuracy_score(y_test, test_preds), 2)*100

    conf_matrix = confusion_matrix(y_test, test_preds, as_frame=True, normalised=False)
    report = classification_report(y_test, test_preds)

    if SHOW:
        print('-'*5 + ' Evaluation of Performance ' + '-'*5)
        print(f'Training Accuracy: {train_acc}%')
        print(f'Validation Accuracy (during 5-fold CV): {val_acc}%')
        print(f'Test Accuracy: {test_acc}%'); print();

        print(conf_matrix)
        print(report)

        if input('SAVE? (y/n)' ) == 'y':
            generate_summary(filepath = './data/results', name='sklearn_random_forest', 
                             best_n_estimators = best_n_estimators,
                             best_criterion = best_criterion,
                             best_max_depth = best_max_depth,
                             best_bootstrap = best_bootstrap,
                             training_accuracy = train_acc,
                             validation_accuracy = val_acc,
                             test_accuracy = test_acc,
                             confusion_matrix = conf_matrix,
                             classification_report = report)

if __name__ == '__main__':
    run_sklearn_random_forest()

