import os
import sys

sys.path.insert(0, os.path.abspath(''))

# external libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# custom imports
from scripts.models import NeuralNetworkClassifier
from scripts.models.neural_net import DenseLayer

from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions

sns.set_style('darkgrid')
np.random.seed(1)
SHOW = True

def main():
    X, y = load_iris(return_X_y=True)

    uniq_idx = np.unique(X, return_index=True, axis=0)[1]
    X = X[uniq_idx]
    y = y[uniq_idx]

    scaler = StandardScaler()  
    x = scaler.fit_transform(X)

    clf = NeuralNetworkClassifier(
            layers = [DenseLayer(n_in=4, n_out=20, activation='relu', name='fc1'),
                      DenseLayer(n_in=20, n_out=3, activation='softmax', name='output')],
            loss='cross_entropy', 
            name=f'Simple NN'
            )

    clf.fit(X, y, epochs=100, lr=0.01, num_batches=10, verbose=1)

    loss_history = clf.loss_history
    acc_history = clf.accuracy_history

    fig, ax = plt.subplots(ncols=2,figsize=(8, 3))
    sns.lineplot(range(len(loss_history)), loss_history, ax=ax[0], color='blue', label='Loss History')
    ax[0].set_title('Loss History'), 
    sns.lineplot(range(len(acc_history)), acc_history, ax=ax[1], color='orange', label='Training Accuracy')
    ax[1].set_title('Training Accuracy History'), 

    if SHOW:
        plt.show()

        if input('SAVE? (y/n): ') == 'y':
            fig.savefig('./data/figures/assert_nn_overfit.pdf')
            print('Saved figure to ./data/figures/assert_nn_overfit.pdf')

if __name__ == '__main__':
    main()
