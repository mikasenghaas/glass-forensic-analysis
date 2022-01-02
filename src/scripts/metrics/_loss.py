"""
Module that contains commonly used loss-functions in machine learning. The following
functions are defined:

- :func:`se`
- :func:`ae`
- :func:`zero_one_loss`
- :func:`cross_entropy`
"""

import numpy as np

def se(y, p):
    """Squared error.

    Sqaured error can be defined as follows:

    .. math::

        \sum_i^n (y_i - p_i)^2
    
    where :math:`n` is the number of provided records.

    Parameters
    ----------
    y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    p : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    float
        Squared error as desribed above.
    
    Notes
    -----
    Usually used for regression problems.
    """
    return np.sum((y - p)**2)

def ae(y, p):

    """Absolute error.

    Absolute error can be defined as follows:

    .. math::

        \sum_i^n abs(y_i - p_i)
    
    where :math:`n` is the number of provided records.

    Parameters
    ----------
    y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    p : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    float
        Absolute error as desribed above.
    """
    return np.abs(y-p).sum()

def zero_one_loss(y, p):
    """Number of incorrectly classified records.

    Parameters
    ----------
    y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    p : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    int
        Number of misclassified records.
    """
    return np.sum(y != p)

def cross_entropy(y, p):

    """
    Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events.
    (`source <https://machinelearningmastery.com/cross-entropy-for-machine-learning/>`_)

    Parameters
    ----------
    y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    p : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    float
        Cross entropy score.
    """

    return -np.sum(np.log(p) * y)
