import os
import sys

sys.path.insert(0, os.path.abspath(''))

# external libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# custom imports
from scripts.models import NeuralNetworkClassifier
from scripts.models.neural_net import DenseLayer

from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions


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

    scaler = StandardScaler()  
    iris_X = scaler.fit_transform(iris_X)
    moons_X = scaler.fit_transform(moons_X)
    circles_X = scaler.fit_transform(circles_X)

    data = {'iris': [iris_X, iris_y],
            'moons': [moons_X, moons_y],
            'circles': [circles_X, circles_y]}

    # ----- plotting --------
    epochs = [50, 100, 200, 500]

    print('starting training')
    fig, axes = plt.subplots(nrows = len(data), ncols = len(epochs), figsize = (10*len(epochs), 4*len(data)))
    for i, dataset in enumerate(data.keys()):
        X, y = data[dataset]
        for j in range(len(epochs)):
            clf = NeuralNetworkClassifier(
                    layers = [DenseLayer(n_in=2, n_out=10, activation='relu', name='fc1')],
                    loss='cross_entropy', 
                    name=f'Simple NN'
                    )

            clf.fit(X, y, epochs=epochs[j], lr=0.2, batch_size=1, verbose=1)
            plot_2d_decision_regions(X, y, clf, ax=axes[i][j], title=f'NN (Epochs: {epochs[j]})')

            if j == 0:
                axes[i][j].set_ylabel(f'{dataset.title()}')
    fig.tight_layout()

    if SHOW:
        plt.show()

    if SAVE:
        fig.savefig(f'{SAVEPATH}/custom_nn.pdf')
        print(f'Saved PDF to {SAVEPATH}')


    """
    # ------ 100% acuracy on iris ------
    X, y = load_iris(return_X_y=True)
    clf = NeuralNetworkClassifier(
            layers = [DenseLayer(n_in=4, n_out=10, activation='relu', name='fc1')],
            loss='cross_entropy', 
            name=f'Test'
            )

    clf.fit(X, y, batch_size=2**5, epochs=500, lr=0.01, verbose=1)
    clf.score()
    """

if __name__ == '__main__':
    main()
