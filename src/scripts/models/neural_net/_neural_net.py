import numpy as np
from tqdm import tqdm
from icecream import ic
from sklearn.preprocessing import OneHotEncoder

from timeit import default_timer

from ...base import BaseClassifier
from ...utils import validate_feature_matrix, validate_target_vector
from ...metrics import se, mse, cross_entropy, accuracy_score
from ._autograd import Var
from ._dense_layer import DenseLayer
from ._helper import convert_to_var, softmax, hot_encode

class NeuralNetworkClassifier(BaseClassifier):
    def __init__(self, layers=[], loss='squared_error', name='NeuralNetworkClassifier'):
        super().__init__()

        self.name = name

        self.X = self.y = self.n = self.p = None
        self.y_hot = None
        self.fitted = False

        self.layers = layers
        self.parameters = self._parameters()

        if loss == 'cross_entropy':
            self.loss = cross_entropy
        elif loss == 'squared_error':
            self.loss = se
        elif loss == 'mean_squared_error':
            self.loss = mse
        else:
            raise NotImplementedError(f"{loss} not yet implemented. Choose from ['cross_entropy', 'squared_error', 'mean_squared_error']") 

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        """
        Computes the forward pass of the MLP: x = layer(x) for each layer in layers
        """
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def fit(self, X, y, num_batches=1, epochs=1000, lr = 0.01, save_at=None, verbose=0):
        self.X = validate_feature_matrix(X)
        self.X = convert_to_var(self.X)
        self.y = y 

        self.n, self.p = self.X.shape
        self.k = len(np.unique(y))
        
        assert self.layers[-1].neurons() == self.k, 'Output Layer must have out dimension equivalent to the number of classes'

        # populate label-intcode dictionaries
        unique_classes = np.unique(y)

        self.label = {k: unique_classes[k] for k in range(self.k)}
        self.intcode = {unique_classes[k]:k for k in range(self.k)}

        # one hot encode y (into nxk matrix of Vars)
        y_hot = hot_encode(self.y, self.intcode)
        self.y_hot = convert_to_var(y_hot)

        # compute batch size
        if isinstance(num_batches, float):
            num_batches = int(X.shape[0] * num_batches)
        elif isinstance(num_batches, int):
            num_batches = num_batches
        else:
            assert False, 'wrong type for batch size'

        if not save_at is None:
            assert isinstance(save_at, list), 'save_at must be of type list'
            assert all(save_at) <= epochs and all(save_at) > 0, 'all values in save_at must be at most epochs and greater 0'
            save_at = set(save_at)

        # training loop
        self.loss_history = []
        self.accuracy_history = []
        self.saves = []
        idx = np.arange(self.n)
        for epoch in range(epochs):
            start_epoch = default_timer()
            
            np.random.shuffle(idx) 
            batch_idxs = np.array_split(idx, num_batches)

            for batch_idx in batch_idxs:
                X_batch = self.X[batch_idx]
                y_batch = self.y_hot[batch_idx]

                """
                if num_batches < self.n :
                    batch_idxs = np.random.choice(list(range(self.n)), num_batches, replace=False) 
                    X_batch = self.X[batch_idxs]
                    y_batch = self.y_hot[batch_idxs]
                else: 
                    X_batch = self.X
                    y_batch = self.y_hot
                """

                # get the probabilities for each datapoint of belonging to each class
                probs = self.forward(X_batch) # num_batches_n x k matrix of probs for each data point for each class 

                # compute loss on one-hot encoded target matrix
                assert not np.any(probs < Var(0.0)), 'probs must be > 0, due to softmax'
                loss = self.loss(y_batch, probs) / Var(len(y_batch))


                # zeroing out gradients
                for param in self.parameters:
                    param.grad = 0.0

                # backward prop 
                loss.backward() # 0.33/0.66 = 50% of epoch time

                # update weights
                for param in self.parameters: 
                    param.v -= lr * param.grad

            # append loss to training history
            self.loss_history.append(loss.v)

            # compute training accuracy
            preds = self.predict(self.X)
            training_accuracy = accuracy_score(self.y, preds) # 20% of epoch time
            self.accuracy_history.append(training_accuracy) 

            if save_at:
                if epoch in save_at:
                    print('Saved.')
                    self.saves.append(preds)

            if verbose:
                if epoch % verbose == 0:
                    end_epoch = default_timer() - start_epoch
                    print(f'> Epoch: {epoch} - #Batches: {num_batches} - Time: {round(end_epoch, 2)}s - '\
                            f'Loss: {loss.v} - Training Accuracy: {round(training_accuracy, 2)}')


        self.fitted = True

    def predict(self, X):
        probs = self.predict_proba(X)

        return np.array([self.label[pred] for pred in np.argmax(probs, axis=1).astype(int)])

    def predict_proba(self, X):
        X = convert_to_var(X)

        return self.forward(X) # probs for classes

    def _parameters(self):
        """ Returns all the parameters of the layers as a 1d np array"""
        return np.hstack([layer.parameters() for layer in self.layers])

    def _total_parameters(self):
        return len(self.parameters)

    def summary(self):
        s = f"Name: {self.name}\n\n" 
        header = 'Layer\t\tWeight Dim\tBias Dim\tTotal Parameters\n'
        s += header
        s += '=' * (len(header)+20) + '\n'
        for layer in self.layers:
            s += repr(layer) + '\n'
        s += '=' * (len(header)+20) + '\n'
        s += f'\t\t\t\t\t\t{self._total_parameters()}'
        return s
