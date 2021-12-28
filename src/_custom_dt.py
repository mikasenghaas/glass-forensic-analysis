# external libraries
import numpy as np
from matplotlib import pyplot as plt

# custom imports
from scripts.models import DecisionTreeClassifier

from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions
from scripts.utils import get_data

np.random.seed(1)

def main():
    # ------ loading and preprocessing ------
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=True)

    # ------ constructing model ------

    # initialise and train model
    clf = DecisionTreeClassifier(max_depth=4) # most generalising; can achieve 1.0 accuracy for depth >= 8
    clf.fit(X_train, y_train)

    # get predictions for training and test split
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    # evaluate performance
    print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
    print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}')

    print(confusion_matrix(y_test, test_preds, as_frame=True))

if __name__ == '__main__':
    main()

