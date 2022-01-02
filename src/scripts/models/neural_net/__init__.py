"""
The Neural Network module contains the following three core classes:

- :class:`Var`
- :class:`DenseLayer`
- :class:`NeuralNetworkClassifier`

These classes combined allow for training a neural network on any dataset.
"""

from ._autograd import Var
from ._dense_layer import DenseLayer
from ._neural_net import NeuralNetworkClassifier

__all__ = [
        'Var',
        'DenseLayer',
        'NeuralNetworkClassifier'
        ]
