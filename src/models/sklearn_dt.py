import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# external libraries
import numpy as np
import pandas as pd
import graphviz
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree, export_graphviz
from sklearn.metrics import classification_report

# custom imports
from scripts.plotting import plot_2d_decision_regions
from scripts.metrics import accuracy_score, confusion_matrix
from scripts.utils import get_data

DO_PCA = True
POLYNOMIAL_FEATURES = False

SHOW = False
SAVE = False
np.random.seed(1)

def main():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=False)

    if POLYNOMIAL_FEATURES:
        engineer = PolynomialFeatures(degree = 3) 
        X_train = engineer.fit_transform(X_train)
        X_test = engineer.fit_transform(X_test)

    # ------ constructing model ------
    scaler = StandardScaler()
    scaler.fit(X_train)

    if DO_PCA:
        pca = PCA()
        pca.fit(X_train)
    
        pipe = Pipeline(steps=[('scaler', scaler),
                               ('pca', PCA()),
                               ('decision_tree', DecisionTreeClassifier(splitter='best', max_features=None))])

        # define hyper parameters to grid search
        params = {
                'pca__n_components': list(range(1, X_train.shape[1]+1)),
                'decision_tree__criterion': ['gini', 'entropy'],
                'decision_tree__max_depth': list(range(1, 10)),
                }

    else:
        pipe = Pipeline(steps=[('scaler', scaler),
                               ('decision_tree', DecisionTreeClassifier(splitter='best', max_features=None, random_state=1))])

        # define hyper parameters to grid search
        params = {
                'decision_tree__criterion': ['gini', 'entropy'],
                'decision_tree__max_depth': list(range(1, 10)),
                }


    # define and train on grid
    grid = GridSearchCV(pipe, params, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # cv results
    print('\n' + '-'*5 + ' CV Results ' + '-'*5)
    print(pd.DataFrame(grid.cv_results_))

    # report back best combination of hyperparameters
    print('-'*5 + ' Best Hyperparameters ' + '-'*5)
    print('Best Criterion:', grid.best_estimator_.get_params()['decision_tree__criterion'])
    print('Best Max Depth:', grid.best_estimator_.get_params()['decision_tree__max_depth'])
    if DO_PCA:
        print('Best Number Of Components:', grid.best_estimator_.get_params()['pca__n_components']); print()

    # final model
    clf = grid.best_estimator_


    # evaluate performance
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    print('-'*5 + ' Evaluation of Performance ' + '-'*5)
    print(f'Training Accuracy: {round(accuracy_score(y_train, train_preds), 2)*100}%')
    print(f'Validation Accuracy (during 5-fold CV): {round(grid.best_score_, 2)*100}%')
    print(f'Test Accuracy: {round(accuracy_score(y_test, test_preds), 2)*100}%'); print();

    print(confusion_matrix(y_test, test_preds, as_frame=True, normalised=False))
    print(classification_report(y_test, test_preds))


    # graphviz plotting
    if DO_PCA:
        FEATURE_NAMES = [f'PC {i}' for i in range(grid.best_estimator_.get_params()['pca__n_components'])]
    else:
        FEATURE_NAMES = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    CLASS_NAMES = ['Window from Building (float-processed)',
                   'Window from Building (non-float processed)',
                   'Window from Vehicle',
                   'Container',
                   'Tableware',
                   'Headlamp']

    # fig = plt.figure(figsize=(25, 20))
    # _ = plot_tree(clf['decision_tree'], feature_names=FEATURE_NAMES, class_names=CLASS_NAMES, filled=True)

    dot_data = export_graphviz(clf['decision_tree'], out_file=None,
                               feature_names=FEATURE_NAMES,
                               class_names=CLASS_NAMES,
                               filled=True)

    graph = graphviz.Source(dot_data, format="png")


    if SHOW:
        print('show')
        plt.show()

    if SAVE: 
        SAVEPATH = './data/figures'
        FILENAME = 'graphviz_sklearn_dt'

        if DO_PCA:
            FILENAME += '_pca'

        graph.render(f'{SAVEPATH}/{FILENAME}')
        print('saved')


if __name__ == '__main__':
    main()
