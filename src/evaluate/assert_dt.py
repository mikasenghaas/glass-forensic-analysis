import os
import sys

sys.path.insert(0, os.path.abspath(''))

# external libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.model_selection import train_test_split

# custom imports
from scripts.models import DecisionTreeClassifier

from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions

sns.set_style('darkgrid')

# global configs
np.random.seed(1)
SAVE = True
SHOW = True

def main():
    # ------ loading and preprocessing data ------
    iris_X, iris_y = load_iris(return_X_y=True)
    iris_X = iris_X[:, :2]

    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=.3)

    fig, ax = plt.subplots(figsize=(8, 6))

    depths = list(range(1, 16))
    train_acc = []
    test_acc = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d)
        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        train_acc.append(accuracy_score(y_train, train_pred))
        test_acc.append(accuracy_score(y_test, test_pred))

    sns.lineplot(depths, train_acc, ax=ax, color='blue', label='Training Accuracy')
    sns.lineplot(depths, test_acc, ax=ax, color='red', label='Test Accuracy')
    ax.set_title('Training Accuracy on different depths of the tree')
    ax.set_xlabel('Max Depth of Tree')
    ax.legend(loc='best')

    if SHOW:
        plt.show()

    if input('SAVE? (y/n): ') == 'y':
        SAVEPATH = './data/figures'
        fig.savefig(f'{SAVEPATH}/assert_dt_overfit.pdf')
        print(f'Saved PDF to {SAVEPATH}/assert_dt_overfit.pdf')

if __name__ == '__main__':
    main()
