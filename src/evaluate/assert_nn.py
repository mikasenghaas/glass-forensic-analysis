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

np.random.seed(1)
SHOW = True

def main():
    X, y = load_iris(return_X_y=True)

    scaler = StandardScaler()  
    x = scaler.fit_transform(X)

    clf = NeuralNetworkClassifier(
            layers = [DenseLayer(n_in=4, n_out=20, activation='relu', name='fc1')],
            loss='cross_entropy', 
            name=f'Simple NN'
            )

    clf.fit(X, y, epochs=100, lr=0.01, batch_size=10, verbose=1)

    loss_history = clf.loss_history
    acc_history = clf.accuracy_history

    fig, ax = plt.subplots(ncols=2,figsize=(8, 3))
    ax[0].plot(range(len(loss_history)), loss_history), 
    ax[0].set_title('Loss History'), 
    ax[1].plot(range(len(acc_history)), acc_history)
    ax[1].set_title('Training Accuracy History'), 

    if SHOW:
        plt.show()

        if input('SAVE? (y/n): ') == 'y':
            fig.savefig('./data/results/assert_nn_overfit.pdf')
            print('Saved figure to ./data/results/assert_nn_overfit.pdf')

if __name__ == '__main__':
    main()
