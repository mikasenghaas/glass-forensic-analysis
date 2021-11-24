import numpy as np
from icecream import ic

from ...utils import validate_feature_matrix, validate_target_vector
from ._autograd import Var
from ._dense_layer import DenseLayer
from ._helper import convert_to_var

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.X = self.y = self.n = self.p = None
        self.fitted = False

        self.parameters = self._parameters()

    def forward(self, X):
        """
        Computes the forward pass of the MLP: x = layer(x) for each layer in layers
        """
        X = X.reshape(-1, 1)

        for layer in self.layers:
            X = layer.forward(X)

        return X

    def fit(self, X, y, batch_size=1, epochs=100, lr = 0.01):
        self.X = validate_feature_matrix(X)
        self.X = convert_to_var(self.X)

        self.y = validate_target_vector(y)
        self.y = convert_to_var(self.y)

        self.n, self.p = self.X.shape
        batch_size = int(X.shape[0] * batch_size)
        #batch_idxs = np.random.choice(list(range(self.n)), batch_size, replace=False) 

        # training loop
        self.training_history = []
        for epoch in range(epochs):
            if batch_size < self.n :
                batch_idxs = np.random.choice(list(range(self.n)), batch_size, replace=False) 
                X_batch = self.X[batch_idxs]
                y_batch = self.y[batch_idxs]
            else: 
                X_batch = self.X
                y_batch = self.y

            probs = self.forward(X_batch) # n x k matrix of probs for each data point for each class 
            
            preds = np.argmax(probs, axis=1).astype(int)
            print(probs.shape)
            print(preds.shape)
            updated_probs = np.take_along_axis(probs, preds[:, None], axis=1)
            print(updated_probs.shape)

            loss = np.sum((y_batch - updated_probs)**2) / Var(batch_size)

            self.training_history.append(loss.v)




            """
            for idx in batch_idxs: 
                X = self.X[idx]
                y = self.y[idx]

                X = np.array([Var(float(x)) for x in X])
                y = Var(int(y))

                pred = self.forward(X).flatten() # probs for classes
                ind = int(np.argmax([p.v for p in pred])) # max prob index

                loss += (y - pred[ind])**2


            loss = loss / Var(batch_size) 
            """



            # zeroing out gradients
            for param in self.parameters:
                param.grad = 0.0

            # recompute gradients
            loss.backward()

            # update weights
            for param in self.parameters: 
                param.v -= lr * param.grad

            if epoch % 10 == 0:
                print(epoch, loss)



            """
            if batch_size < self.n :
                batch_idxs = np.random.choice(list(range(self.n)), batch_size, replace=False) 
                X_batch = self.X[batch_idxs]
                y_batch = self.y[batch_idxs]
            else: 
                X_batch = self.X
                y_batch = self.y
            """


        self.fitted = True

    def predict(self, X):
        X = convert_to_var(X)
        preds = []
        for x in X:
            pred = self.forward(np.array(x)).flatten() # probs for classes
            ind = int(np.argmax([p.v for p in pred])) # max prob index

            preds.append(ind)

        return preds

    def _parameters(self):
        """ Returns all the parameters of the layers as a 1d np array"""
        return np.hstack([layer.parameters() for layer in self.layers])

