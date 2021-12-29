# external libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC 
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

    # ------ constructing model ------

    # define pipeline (scaling, pca, decision tree)
    scaler = StandardScaler()
    scaler.fit(X_train)
    pipe = Pipeline(steps=[('scaler', scaler),
                           ('pca', PCA()),
                           ('svm', SVC())])

    # define hyper parameters to grid search
    params = {
            'pca__n_components': list(range(1, X_train.shape[1]+1)),
            'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svm__C': list(np.linspace(0.2, 1, 4))
            }

    # define and train on grid
    grid = GridSearchCV(pipe, params)
    grid.fit(X_train, y_train)

    # report back best combination of hyperparameters
    print('Best Kernel:', grid.best_estimator_.get_params()['svm__kernel'])
    print('Best Regulation Parameter C:', grid.best_estimator_.get_params()['svm__C'])
    print('Best Number Of Components:', grid.best_estimator_.get_params()['pca__n_components']); print();

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
