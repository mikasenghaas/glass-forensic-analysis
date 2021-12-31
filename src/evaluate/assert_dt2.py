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
SAVEPATH = './data/figures'

def main():
    # ------ loading and preprocessing data ------
    iris_X, iris_y = load_iris(return_X_y=True)
    iris_X = iris_X[:, :2]
    moons_X, moons_y = make_moons(random_state=1)
    circles_X, circles_y = make_circles(random_state=1)

    data = {'iris': [iris_X, iris_y],
            'moons': [moons_X, moons_y],
            'circles': [circles_X, circles_y]}

    # ------ constructing models ------
    models = []
    depths = [1, 2, 5, 10, None]
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        models.append(clf)

    # ----- plotting --------
    fig, axes = plt.subplots(nrows = len(data), ncols = len(models), figsize = (10*len(models), 4*len(data)))
    for i, dataset in enumerate(data.keys()):
        X, y = data[dataset]
        for j in range(len(models)):
            model = models[j]
            model.fit(X, y)
            plot_2d_decision_regions(X, y, model, ax=axes[i][j], title=f'DT (max_depth={depths[j]})')

            if j == 0:
                axes[i][j].set_ylabel(f'{dataset.title()}')
    fig.tight_layout()

    if SHOW:
        plt.show()

    if SAVE:
        fig.savefig(f'{SAVEPATH}/dt_correctness.pdf')
        print(f'Saved PDF to {SAVEPATH}')

if __name__ == '__main__':
    main()
