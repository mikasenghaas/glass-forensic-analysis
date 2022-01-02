from math import tanh, log, log10, exp
import numpy as np 

class Var:
    """
    A variable which holds a number and enables gradient computations.

    Adapted from Rasmus Berg Palm `repository <https://github.com/rasmusbergpalm/nanograd>`_.

    Parameters
    ----------
    val : float or int
        The actual value of the number.
    parents : list, optional
        List where each item has the following structure (:class:`Var`, gradient)
    
    Attributes
    ----------
    grad : float or int
        Partial derivative of this variable with respect to the variable from which the bakcward method was called.
    
    Raises
    ------
    AssertionError
        If the val is not float or int.

    """

    def __init__(self, val, parents=None):
        assert type(val) in {float, int}
        if parents is None:
            parents = []
        self.v = val
        self.parents = parents
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for parent, grad in self.parents:
            parent.backprop(grad * bp)

    def backward(self):
        """
        Compute gradient of this Var with respect to all its parents.
        """
        self.backprop(1.0)

    def __add__(self, other):
        return Var(self.v + other.v, [(self, 1.0), (other, 1.0)])

    def __mul__(self, other): 
        return Var(self.v * other.v, [(self, other.v), (other, self.v)])

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Var(self.v ** power, [(self, power * self.v ** (power - 1))])

    def __neg__(self):
        return Var(-1.0) * self

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __lt__(self, other):
        return self.v < other.v 

    def tanh(self):
        """Peform Tanh activation function.
        """
        return Var(tanh(self.v), [(self, 1 - tanh(self.v) ** 2)])

    def relu(self):
        """Peform Relu activation function.

        Relu is defined as follows:

        .. math::
        
            relu(x) = max(0, x)
        
        """
        return Var(self.v if self.v > 0.0 else 0.0, [(self, 1.0 if self.v > 0.0 else 0.0)])

    def log(self):
        """Peform log function. (base 10)
        """
        return Var(log10(self.v), [(self, 1 / self.v * log(10))]) 

    def exp(self):
        """Peform exp function.
        """
        return Var(exp(self.v), [(self, exp(self.v))])

    def __float__(self):
        return self.v

    def __int__(self):
        return int(self.v)

    def __repr__(self):
        return f'{self.v}'
