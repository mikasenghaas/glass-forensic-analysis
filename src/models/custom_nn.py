import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from scripts.models import NeuralNetworkClassifier
from scripts.models.neural_net import DenseLayer

from scripts.metrics import accuracy_score, confusion_matrix
from scripts.plotting import plot_2d_decision_regions
from scripts.utils import get_data, generate_summary

# global settings
sns.set_style('darkgrid')
np.random.seed(1)
SHOW = True

def main():
    # ------ loading and preprocessing ------

    # load and split data
    X_train, X_test, y_train, y_test = get_data(raw=True, scaled=True)

    #------ constructing model ------
    nn = NeuralNetworkClassifier(
            layers = [DenseLayer(n_in=9, n_out=20, activation='relu', name='fc1'),
                      DenseLayer(n_in=20, n_out=6, activation='softmax', name='output')],
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
    lr = 0.01

    # fit model
    nn.fit(X_train, y_train, batch_size=10, verbose=1, epochs=epochs, lr=lr)

    # plot training history
    # plot training/ validation accuracy and loss history
    fig, ax = plt.subplots(ncols=2, figsize=(8,3))
    history = {'loss': nn.loss_history, 'accuracy': nn.accuracy_history}
    for i, title in zip(range(2), ['loss', 'accuracy']):
        #ax[i].plot(list(range(1, epochs+1)), history[title], label=f'Training {title.title()}')
        sns.lineplot(range(1, epochs+1), history[title], label=f'Training {title.title()}', ax=ax[i])
        ax[i].set_title(f'History of {title.title()}')
        ax[i].legend(loc='best')

    if SHOW:
        plt.show()

        if input('SAVE? (y/n)' ) == 'y':
            fig.savefig('./data/figures/custom_nn_training.pdf')
            print('saved')


    # get predictions for training and test split
    train_preds = nn.predict(X_train)
    test_preds = nn.predict(X_test)

    # evaluate performance
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    conf_matrix = confusion_matrix(y_test, test_preds, as_frame=True, normalised=False)
    report = classification_report(y_test, test_preds)

    if SHOW:
        print(f'Final Train Accuracy: {train_acc}\n')
        print(f'Final Test Accuracy: {test_acc}\n')

        print(conf_matrix)
        print(report)

        if input('SAVE? (y/n)' ) == 'y':
            generate_summary(filepath = './data/results', name='custom_neural_network', 
                             training_accuracy = train_acc,
                             test_accuracy = test_acc,
                             confusion_matrix = conf_matrix,
                             classification_report = report)

if __name__ == '__main__':
    main()
