import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions

from scripts.models.neural_net import DenseLayer
from scripts.models import NeuralNetworkClassifier
from scripts.metrics import accuracy_score

from scripts.plotting import plot_2d_decision_regions

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# iris data 
X, y = load_iris(return_X_y=True)
X = X[:, :2]
#y = y[y!=2]

# scale features for gradient descent to work properly
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train test split to evaluate out-of-bag-performance
X_train, X_test, y_train, y_test = train_test_split(X, y)

dl = DenseLayer(n_in=2, n_out=5, activation='tanh', name='fc1')
#dl2 = DenseLayer(n_in=10, n_out=5, activation='tanh', name='fc2')
#dl3 = DenseLayer(n_in=5, n_out=4, activation='tanh', name='fc3')

nn = NeuralNetworkClassifier(layers = [dl], loss='cross_entropy', name='TestNeuralNetClassifier')

"""
Alternative Initialisation:
    nn = NeuralNetworkClassifier(name='TestNeuralNetClassifier')
    nn.add(DenseLayer(n_features, 5, activation='tanh', name='fc1')
    nn.add(DenseLayer(5, 4, activation='tanh', name='fc2')
"""


#  train model
epochs = 50
lr = 0.005
nn.fit(X_train, y_train, batch_size=1, epochs=epochs, lr=lr)

# plot training history
plt.plot(list(range(epochs)), nn.training_history)
plt.show()

# method to print overview of neural net
print(nn.summary())

# get predictions for training and test split
train_preds = nn.predict(X_train)
test_preds = nn.predict(X_test)

# evaluate performance
print(f'Training Accuracy: {accuracy_score(y_train, train_preds)}')
print(f'Test Accuracy: {accuracy_score(y_test, test_preds)}')

fig = plot_2d_decision_regions(X_train, y_train, nn)
plt.show()

"""
BENCHMARK: LogisticRegression 
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_train)
print(accuracy_score(y_train, pred))
"""
