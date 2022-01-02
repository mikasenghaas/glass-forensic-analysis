import os
import sys

sys.path.insert(0, os.path.abspath('.'))

# external libraries
import numpy as np
import pandas as pd
import graphviz
import pydotplus
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
from scripts.utils import get_data, generate_summary

# global configs
np.random.seed(1)
DO_PCA = False
POLYNOMIAL_FEATURES = False
SHOW = True

def run_sklearn_dt():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=False)

    if POLYNOMIAL_FEATURES:
        engineer = PolynomialFeatures(degree = 3) 
        X_train = engineer.fit_transform(X_train)
        X_test = engineer.fit_transform(X_test)

    scaler = StandardScaler()
    scaler.fit(X_train)

    if DO_PCA:
        pca = PCA()
        pca.fit(X_train)

    # ------ constructing model ------

    if DO_PCA:
        pipe = Pipeline(steps=[('scaler', scaler),
                               ('pca', pca),
                               ('decision_tree', DecisionTreeClassifier(random_state=1))])

        # define hyper parameters to grid search
        params = {
                'pca__n_components': list(range(1, X_train.shape[1]+1)),
                'decision_tree__criterion': ['gini', 'entropy'],
                'decision_tree__max_depth': list(range(1, 10)),
                'decision_tree__splitter': ['best', 'random'] 
                }

    else:
        pipe = Pipeline(steps=[('scaler', scaler),
                               ('decision_tree', DecisionTreeClassifier(random_state=1))])

        # define hyper parameters to grid search
        params = {
                'decision_tree__criterion': ['gini', 'entropy'],
                'decision_tree__max_depth': list(range(1, 10)),
                'decision_tree__max_features': list(range(1, 10)),
                'decision_tree__splitter': ['best', 'random'] 
                }


    # define and train on grid
    grid = GridSearchCV(pipe, params, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)


    # ------ report best hyper parameters ------

    # report back best combination of hyperparameters
    print('-'*5 + ' Best Hyperparameters ' + '-'*5)
    best_criterion = grid.best_estimator_.get_params()['decision_tree__criterion']
    best_max_depth = grid.best_estimator_.get_params()['decision_tree__max_depth']
    best_splitter = grid.best_estimator_.get_params()['decision_tree__splitter']
    if DO_PCA:
        best_n_components = grid.best_estimator_.get_params()['pca__n_components']
    else:
        best_max_features = grid.best_estimator_.get_params()['decision_tree__max_features']



    if SHOW:
        print('Best Criterion: ', best_criterion)
        print('Best Max Depth: ', best_max_depth)
        print('Best Splitter: ', best_splitter)
        if DO_PCA:
            print('Best PCA Components:', best_n_components)
        else: 
            print('Best Max Features:', best_max_features)

    # final model
    clf = grid.best_estimator_

    # ------ evaluate performance  ------

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    train_acc = round(accuracy_score(y_train, train_preds), 2)*100
    val_acc = round(grid.best_score_, 2)*100
    test_acc = round(accuracy_score(y_test, test_preds), 2)*100

    conf_matrix = confusion_matrix(y_test, test_preds, as_frame=True, normalised=False)
    report = classification_report(y_test, test_preds)

    # ------ show and save results ------

    if SHOW:
        print('-'*5 + ' Evaluation of Performance ' + '-'*5)
        print(f'Training Accuracy: {train_acc}%')
        print(f'Validation Accuracy (during 5-fold CV): {val_acc}%')
        print(f'Test Accuracy: {test_acc}%'); print();

        print(conf_matrix)
        print(report)

        if input('SAVE? (y/n)' ) == 'y':
            generate_summary(filepath = './data/results', name='sklearn_dt', 
                             best_criterion = best_criterion,
                             best_max_depth = best_max_depth,
                             best_splitter = best_splitter,
                             best_max_features = best_max_features,
                             training_accuracy = train_acc,
                             validation_accuracy = val_acc,
                             test_accuracy = test_acc,
                             confusion_matrix = conf_matrix,
                             classification_report = report)

    # ------ show and save decision tree visualisation ------

    if input('Plot DT? (y/n)') == 'y':
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

        dot_data = export_graphviz(clf['decision_tree'], out_file=None,
                                   feature_names=FEATURE_NAMES,
                                   class_names=CLASS_NAMES,
                                   filled=True,
                                   rounded=True)
        graph = graphviz.Source(dot_data, format="png")
        #pydot_graph = pydotplus.graph_from_dot_data(dot_data)
        #pydot_graph.write_png('original_tree.png')
        #pydot_graph.set_size('"1000,500!"')
        # graph.set_size('"10,5!"')

        if SHOW:
            plt.show()
            if input('SAVE? (y/n)' ) == 'y':
                SAVEPATH = './data/figures'
                FILENAME = 'graphviz_sklearn_dt'

                if DO_PCA:
                    FILENAME += '_pca'

                #pydot_graph.write_png(f'{SAVEPATH}/{FILENAME}.png')
                graph.render(f'{SAVEPATH}/{FILENAME}')
                print('saved')


if __name__ == '__main__':
    run_sklearn_dt()
