# external libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
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

    # ------ constructing model ------

    # define pipeline (scaling, pca, decision tree)
    scaler = StandardScaler()
    scaler.fit(X_train) # use scaling parameters from training set in pipeline
    pipe = Pipeline(steps=[('scaler', scaler),
                           ('pca', PCA()),
                           ('ensemble', BaggingClassifier(base_estimator=SVC(),
                                                          n_estimators=10, 
                                                          random_state=0))])


    # define hyper parameters to grid search
    params = {
            'pca__n_components': list(range(1, X_train.shape[1]+1)),
            #'ensemble__max_samples' : [0.25, 0.5, 0.75, 1],
            #'ensemble__max_features' : list(range(1, X_train.shape[1])),
            #'ensemble__bootstrap' : [True, False],
            #'base_estimator__svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #'base_estimator__C': list(np.linspace(0.2, 1, 4))
            }

    # define and train on grid
    grid = GridSearchCV(pipe, params)
    grid.fit(X_train, y_train)

    # report back best combination of hyperparameters
    """
    print('Best Criterion:', grid.best_estimator_.get_params()['decision_tree__criterion'])
    print('Best Max Depth:', grid.best_estimator_.get_params()['decision_tree__max_depth'])
    """
    print('Bootstrap Training Sample:', grid.best_estimator_.get_params()['ensemble__bootstrap'])
    print('Bootstrap Training Sample:', grid.best_estimator_.get_params()['ensemble__bootstrap'])
    print('Best Number Of Components:', grid.best_estimator_.get_params()['pca__n_components']); print();

    # final model
    clf = grid.best_estimator_

    # evaluate performance
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
    print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}'); print();

    print(confusion_matrix(y_test, test_preds))
    print(list(range(1, X_train.shape[1]+1)))

if __name__ == '__main__':
    main()
