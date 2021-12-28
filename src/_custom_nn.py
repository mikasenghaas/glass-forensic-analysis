import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions

from scripts.models.neural_net import DenseLayer
from scripts.models import NeuralNetworkClassifier
from scripts.metrics import accuracy_score, confusion_matrix

from scripts.plotting import plot_2d_decision_regions

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(1)

def main():
    """
    # iris data 
    X, y = load_iris(return_X_y=True)
    X = X[:, :2]
    #y = y[y!=2]
    """

    train = np.loadtxt('./data/raw/df_train.csv', skiprows=1, delimiter=',')
    test = np.loadtxt('./data/raw/df_test.csv', skiprows=1, delimiter=',')

    X_train, y_train = train[:, :-1], train[:, -1].astype(int)
    X_test, y_test = test[:, :-1], test[:, -1].astype(int)

    train_n = len(X_train)
    test_n = len(X_test)

    X = np.vstack((X_train, X_test))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test = X[:train_n, :], X[train_n:, :]

    # -----------------
    # construct custom model
    nn = NeuralNetworkClassifier(
            layers = [DenseLayer(n_in=9, n_out=20, activation='relu', name='fc1')],
            loss='cross_entropy', 
            name='CustomModel'
            )

    print(nn.summary())

    # 500 epochs, 0.1 lr, entire batch L, 5 nodes layout
    # new best: 500 epochs, 0.1 lr, 5 batch, 20 nodes layout
    epochs = 1000
    lr = 0.5
    nn.fit(X_train, y_train, batch_size=10, verbose=5, epochs=epochs, lr=lr)

    # plot training history
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    ax[0].plot(list(range(epochs)), nn.loss_history)
    ax[1].plot(list(range(epochs)), nn.accuracy_history)
    plt.show()

    # get predictions for training and test split
    train_preds = nn.predict(X_train)
    print(y_train, train_preds)
    test_preds = nn.predict(X_test)

    # evaluate performance
    print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
    print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}')

    print('\nTraining Confusion Matrix')
    print(confusion_matrix(y_train, train_preds, as_frame=True))
    print('\nTest Confusion Matrix')
    print(confusion_matrix(y_test, test_preds, as_frame=True))

    """
    # plotting (only works with 2 features)
    fig = plot_2d_decision_regions(X_train, y_train, nn)
    plt.show()
    """

if __name__ == '__main__':
    main()
