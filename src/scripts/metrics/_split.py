"""
This module includes functions to quantify quality of split of data for decision tree building.
The following functions are available:

- :func:`gini`
- :func:`entropy`

"""

import numpy as np
from collections import Counter

def gini(y):

    """Compute gini impurity score.

    Gini impurity is usually used within the context of
    DecisionTrees. The value ranges between 0 and 1.
    If 0, it means that within your dataset, you only have one class.
    If more than 0, it means that there is certain likelihood that
    you will misclassify given sample from yout dataset.

    It can be computed using the following formula:

    .. math::
        G(Y) = \sum_{i = 0}^{k} P(i)*(1 - P(i))
    
    where ``k`` is number of classes and :math:`P(i)` is probability of i-th class.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        1d array of labels of classes.
    
    Returns
    -------
    float
        Float between 0 and 1.
    """

    N = len(y)
    counter = Counter(y)

    ans = 0
    for val in counter.values():
        ans += val / N * ( 1 - val / N )
    return ans

def entropy(y):

    """Compute entropy of the given vector.

    Entropy is a measure of disorder. The higher the entropy,
    the more disorder there is present. As an example,
    if you have binary classes where 50 % is positive and the
    rest negative, then your entropy would be 1 (high), if
    you only have positive samples, then your entropy is 0. (low)
    The formula for entropy is as follows:

    .. math::
    
        E(Y) = \sum_i^k -p_i log_2 p_i

    where ``k`` is number of classes you have.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        1d array of labels of classes.
    
    Returns
    -------
    float
        Value between 0 to +inf depending on the number of clasess.
    """

    N = len(y)
    counter = Counter(y)

    ans = 0
    for val in counter.values():
        ans += val / N * np.log(val / N)
    return -ans
