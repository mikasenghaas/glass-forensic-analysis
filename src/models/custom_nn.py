import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from scripts.models import NeuralNetworkClassifier
from scripts.models.neural_net import DenseLayer

from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions
from scripts.utils import get_data

np.random.seed(1)

SHOW = False
SAVE = True

def main():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(raw=False, scaled=True)

    #------ constructing model ------
    nn = NeuralNetworkClassifier(
            layers = [DenseLayer(n_in=9, n_out=20, activation='relu', name='fc1')],
            loss='cross_entropy', 
            name='CustomModel'
            )

    print(nn.summary())

    """
    Current Best Configuration
    --------------------------
    Epochs: 1000, Learning Rate: 0.5, Batch Size: 10, 20 nodes, single hidden layer architecture
    """
    # define hyperparameters
    epochs = 100 
    lr = 0.1

    # fit model
    nn.fit(X_train, y_train, batch_size=10, verbose=1, epochs=epochs, lr=lr)

    # plot training history
    # plot training/ validation accuracy and loss history
    fig, ax = plt.subplots(ncols=2, figsize=(8,3))
    history = {'loss': nn.loss_history, 'accuracy': nn.accuracy_history}
    for i, title in zip(range(2), ['loss', 'accuracy']):
        ax[i].plot(list(range(1, epochs+1)), history[title], c='blue', label=f'Training {title.title()}')
        ax[i].set_title(f'History of {title.title()}')
        ax[i].set_xlabel('#Epochs')
        ax[i].legend(loc='best')

    if SHOW:
        plt.show()
    if SAVE:
        fig.savefig('./data/figures/custom_nn_training_history.pdf')

    # get predictions for training and test split
    train_preds = nn.predict(X_train)
    val_preds = nn.predict(X_val)
    test_preds = nn.predict(X_test)

    # evaluate performance
    print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
    print(f'Validation Accuracy: {accuracy_score(y_val, val_preds)}')
    print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}')

    print(confusion_matrix(y_test, test_preds, as_frame=True, normalised=False))

    print(classification_report(y_test, test_preds))

if __name__ == '__main__':
    main()
