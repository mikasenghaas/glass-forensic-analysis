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
    epochs = [10, 50, 100]

    print('starting training')
    fig, axes = plt.subplots(nrows = len(data), ncols = len(epochs), figsize = (5*len(epochs), 5*len(data)))
    for i, info in enumerate(zip(data.keys(), [3, 2, 2])):
        dataset, k = info
        X, y = data[dataset]
        for j in range(len(epochs)):
            clf = NeuralNetworkClassifier(
                    layers = [DenseLayer(n_in=2, n_out=30, activation='relu', name='fc1'),
                              DenseLayer(n_in=30, n_out=k, activation='softmax', name='output')],
                    loss='cross_entropy', 
                    name=f'Simple NN'
                    )

            clf.fit(X, y, epochs=epochs[j], lr=0.1, batch_size=5, verbose=1)
            plot_2d_decision_regions(X, y, clf, ax=axes[i][j], title=f'NN (Epochs: {epochs[j]})')

            if j == 0:
                axes[i][j].set_ylabel(f'{dataset.title()}')
    fig.tight_layout()

    if SHOW:
        plt.show()

        if input('SAVE? (y/n)') == 'y':
            fig.savefig(f'{SAVEPATH}/assert_nn_toydata.pdf')
            print(f'Saved PDF to {SAVEPATH}/assert_nn_toydata.pdf')


if __name__ == '__main__':
    main()
