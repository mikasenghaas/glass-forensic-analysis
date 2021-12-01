import numpy as np

from ._autograd import Var
from ._helper import convert_to_var, softmax

class DenseLayer:
    def __init__(self, n_in, n_out, activation, name='DenseLayer'):
        """
          n_in: the number of inputs to the layer
          n_out: the number of output neurons in the layer
          act_fn: the non-linear activation function for each neuron
          initializer: The initializer to use to initialize the weights and biases
        """
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
        return len(self.bias)
    
    def dim(self):
        return self.weights.shape, self.bias.shape

    def parameters(self):
        """Returns all the vars of the layer (weights + biases) as a single flat list"""
        return np.hstack((self.weights.flatten(), self.bias))

    def num_params(self):
        return len(self.parameters())
      
    def forward(self, X):
        """ 
        inputs: A n_in length vector of Var's corresponding to the previous layer 
        outputs or the data if it's the first layer.

        Computes the forward pass of the dense layer: For each output neuron, j, 
        it computes: act_fn(weights[i][j]*inputs[i] + bias[j])
        Returns a vector of Vars that is n_out long.
        """
        assert X.shape[1] == self.weights.shape[0], f'Mismatch in second X dimension;'\
                                                    f'tried {X.shape}x{self.weights.shape}'

        #print(f'Passing through {self.name}: {X}x{self.weights} + {self.bias}')
        return self.activation(X @ self.weights + self.bias)
        
    def __repr__(self):    
        return f'{self.name}\t\t{self.weights.shape}\t\t{self.bias.shape}\t\t{self.num_params()}' 
