import os
import sys

sys.path.insert(0, os.path.abspath('.')) # resetting python path to access scripts module

# external libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# custom imports
from scripts.models import DecisionTreeClassifier
from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions
from scripts.utils import get_data, generate_summary

# global configs
np.random.seed(1)
DO_PCA = False
SHOW = True

def main():
    # ------ loading and preprocessing ------
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=True)

    if DO_PCA:
        pca = PCA()
        X_train = pca.fit_transform(X_train)

    # ------ fitting model ------

    # initialise and train model
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=8) # most generalising; can achieve 1.0 accuracy for depth >= 8
    clf.fit(X_train, y_train)

    # ------ evaluate model ------

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    conf_matrix = confusion_matrix(y_test, test_preds, as_frame=True, normalised=False)
    report = classification_report(y_test, test_preds)

    # ------ show and save results ------

    if SHOW:
        print(f'Training Accuracy: {train_acc}')
        print(f'Test Accuracy: {test_acc}')

        print(conf_matrix)
        print(report)

        if input('SAVE? (y/n)' ) == 'y':
            generate_summary(filepath = './data/results', name='custom_dt', 
                 best_criterion = best_criterion,
                 best_max_depth = best_max_depth,
                 training_accuracy = train_acc,
                 validation_accuracy = val_acc,
                 test_accuracy = test_acc,
                 confusion_matrix = conf_matrix,
                 classification_report = report)

             
if __name__ == '__main__':
    main()
