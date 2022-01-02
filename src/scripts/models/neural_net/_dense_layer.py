import numpy as np

from ._autograd import Var
from ._helper import convert_to_var, softmax

class DenseLayer:
    """DenseLayer data structure that works as a building block for the Neural Network class.

    This class serves as a core building block to implement a Neural Network.

    Its main functionality is to take an input X and produce an output Y as a linear combination of the weights and the input (passed through a non linear activation function)

    Parameters
    ----------
    n_in : int 
        The number of inputs that the layer expects (e.g. number of 
        features for the first layer)

    n_out : int
        The number of output neurons 

    activation : str
        The non-linear activation function that is used when forwarding 
        an input through the layer. 
        Possible values are 'relu', 'tanh' or 'softmax' 
   
    name : str, optional
        Name of the layer
    """

    def __init__(self, n_in, n_out, activation, name='DenseLayer'):
        # name of dense layer
        self.name = name

        # randomly initialise weight matrix for dense layer
        self.weights = np.random.rand(n_in, n_out)
        self.weights = convert_to_var(self.weights)

        self.bias = np.random.rand(n_out)
        self.bias = convert_to_var(self.bias)

        # vectorised activation functions
        if activation == 'relu':
            self.activation = np.vectorize(lambda x: x.relu())
        elif activation  == 'tanh':
            self.activation = np.vectorize(lambda x: x.tanh())
        elif activation == 'softmax':
            self.activation = softmax # not working yet
        else:
            raise NotImplementedError("Cannot find Activation Function. Choose from ['relu', 'tanh']")

    def neurons(self):
        """Return the number of neurons

        Returns
        -------
        int
        """
        return len(self.bias)
    
    def dim(self):
        """Returns the dimensions of the weights matrix and the bias vector

        Returns
        -------
        tuple
        """
        return self.weights.shape, self.bias.shape

    def parameters(self):
        """Returns all the vars of the layer (weights + biases) as a 
        single flat list

        Returns
        -------
        1d array
            n x 1 where n is a sum of the number of weights and the number of
            biases
        """

        return np.hstack((self.weights.flatten(), self.bias))

    def num_params(self):
        """Returns the number of parameters

        Returns
        -------
        int

        """
        return len(self.parameters())
      
    def forward(self, X):
        """Computes a forward pass through the layer

        Parameters
        ----------
        X : 2d array of Var instances
            n x n_in where n is the number of provided samples

        Notes
        -----
        Computes the forward pass of the dense layer: For each output neuron, j, 
        it computes: activation(weights[i][j]*inputs[i] + bias[j])

        Returns
        -------
        n x n_out array of Vars where n is the number of provided samples.
        """
        assert X.shape[1] == self.weights.shape[0], f'Mismatch in second X dimension;'\
                                                    f'tried {X.shape}x{self.weights.shape}'

        #print(f'Passing through {self.name}: {X}x{self.weights} + {self.bias}')
        return self.activation(X @ self.weights + self.bias)
        
    def __repr__(self):    
        return f'{self.name}\t\t{self.weights.shape}\t\t{self.bias.shape}\t\t{self.num_params()}' 
