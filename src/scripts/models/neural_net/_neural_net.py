import numpy as np
from tqdm import tqdm
from icecream import ic
from sklearn.preprocessing import OneHotEncoder

from timeit import default_timer

from ...base import BaseClassifier
from ...utils import validate_feature_matrix, validate_target_vector
from ...metrics import se, cross_entropy, accuracy_score
from ._autograd import Var
from ._dense_layer import DenseLayer
from ._helper import convert_to_var, softmax, hot_encode

class NeuralNetworkClassifier(BaseClassifier):
    """NeuralNetworkClassifier 

    This class serves as a high level client facing API. It allows to specify
    all hyper parameters to initialise a feed forward neural network


    Parameters
    ----------
    layers : list
        a list of :class:`DenseLayer`

    loss : str, optional
        loss function to be minimised (default is 'cross_entropy')

    name : str, optional
        Name for the classifier

    Attributes
    ----------
    X : 2d array
        Data points to used to train the neural network

    y : 1d array
        Target classes

    y_hot : 2d array
        One hot encoded target classes

    n : int
        number of data points (X.shape[0])

    p : int
        number of features (X.shape[1])

    fitted : bool
        Boolean to see whether the classifier has been fit

    parameters : 1d array
        array of all the parameters
    """

    def __init__(self, layers=[], loss='cross_entropy', name='NeuralNetworkClassifier'):
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
        else:
            raise NotImplementedError(f"{loss} not yet implemented. Choose from ['cross_entropy', 'squared_error', 'mean_squared_error']") 

    def add(self, layer):
        """Add a dense layer behind the current last dense layer

        layer : :class:`DenseLayer`
        """
        self.layers.append(layer)

    def forward(self, X):
        """Compute the forward pass

        Parameters
        ----------
        X : 2d array
            n x n_in a sample to forward through the network, n_in must 
            correspond to the first :class:`DenseLayer`

        Notes
        -----
        Computes the forward pass of the MLP: x = layer(x) for each 
        layer in layers

        Returns
        -------
        2d array
            n x n_out array where n is the number of data points and n_out 
            corresponds to the number of neurons in the last :class:`DenseLayer`
        """
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def fit(self, X, y, num_batches=1, epochs=1000, lr = 0.01, verbose=0):
        """The training loop of the neural network

        Parameters
        ----------
        X : 2d array
            n x p matrix of data points, where n is the number of data points
            and p the number of features

        y : 1d array
            n x 1 vector of target classes, where n is the number of data points

        num_batches : int, optional
            Number of batches used to train the network in every epoch.
            The parameters are being updated after every forward pass of a batch.
            Default value is 1 which corresponds to feeding the entire data set X
            in every epoch
        
        epochs : int, optional
            Number of training loops 

        lr : float, optional
            learning rate for the gradient descent step

        verbose : int, optional
            Print outs during the training (possible values: 0, 1 or 2)

        Attributes
        ----------
        k : int
            number of unique classes

        loss_history : list
        
        accuracy_history : list
        """

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


        # training loop
        self.loss_history = []
        self.accuracy_history = []
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

            if verbose:
                if epoch % verbose == 0:
                    end_epoch = default_timer() - start_epoch
                    print(f'> Epoch: {epoch} - #Batches: {num_batches} - Time: {round(end_epoch, 2)}s - '\
                            f'Loss: {loss.v} - Training Accuracy: {round(training_accuracy, 2)}')


        self.fitted = True

    def predict(self, X):
        """Predict class labels

        Parameters
        ----------
        X : 2d array
            a sample for which we wish to make predictions
        
        Returns
        -------
        1d array
            predicted labels of size n, where n is the size of the sample X 

        """
        probs = self.predict_proba(X)

        return np.array([self.label[pred] for pred in np.argmax(probs, axis=1).astype(int)])

    def predict_proba(self, X):
        """Predict probabilities for each class

        Parameters
        ----------
        X : 2d array
            a sample for which we wish to make predictions

        Notes
        -----
        For all the data points, find the probabilities of belonging to all the k
        classes

        Returns
        -------
        2d array
            probabilities of size n x k, where n is the size of the sample X and k the number of classes
        """
        X = convert_to_var(X)

        return self.forward(X) # probs for classes

    def _parameters(self):
        """Returns all the parameters of the layers as a 1d np array

        Returns
        -------
        1d array
            n x 1 array, where n is the number of parameters
        """
        return np.hstack([layer.parameters() for layer in self.layers])

    def _total_parameters(self):
        """Number of parameters

        Returns
        -------
        int 
            the number of parameters the Neural Network has
        """
        return len(self.parameters)

    def summary(self):
        """Summary of the neural network

        Returns
        -------
        str
            summary of the network (name,weights,biases,layers,number of parameters)
        """
        s = f"Name: {self.name}\n\n" 
        header = 'Layer\t\tWeight Dim\tBias Dim\tTotal Parameters\n'
        s += header
        s += '=' * (len(header)+20) + '\n'
        for layer in self.layers:
            s += repr(layer) + '\n'
        s += '=' * (len(header)+20) + '\n'
        s += f'\t\t\t\t\t\t{self._total_parameters()}'
        return s
