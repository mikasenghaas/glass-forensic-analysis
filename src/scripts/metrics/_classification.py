"""
This module includes following functions:

- :func:`accuracy_score`
- :func:`classification_error`
- :func:`confusion_matrix`
- :func:`precision_score`
- :func:`recall_score`
- :func:`f1_score`
- :func:`classification_report`

These functions represent metrics and numeric summaries relevant for classification problems.
"""

import numpy as np
import pandas as pd
from ..utils import check_consistent_length

def accuracy_score(y_true, y_pred, normalised=True):
    """Number of correctly classified records.

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_pred : iterale object
        Predicted values.
    normalised : bool, optional
        Normalize the metric by the total number of records.
    
    Returns
    -------
    float or int
        Number of incorrectly classified records.

    Raises
    ------
    :class:`ValueError`
        If the inputted vectors are not of the same length.
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    check_consistent_length(y_true, y_pred)
    n = len(y_true)

    correct = sum(y_true == y_pred)

    if normalised:
        return correct / n
    return correct

def classification_error(y_true, y_pred, normalised=True):
    """Number of incorrectly classified records.

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_pred : iterale object
        Predicted values.
    normalised : bool, optional
        Normalize the metric by the total number of records.
    
    Returns
    -------
    float or int
        Number of incorrectly classified records.

    Raises
    ------
    :class:`ValueError`
        If the inputted vectors are not of the same length.
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    check_consistent_length(y_true, y_pred)
    n = len(y_true)

    misclassified = sum(y_true != y_pred)

    if normalised:
        return misclassified / n 
    return misclassified

def confusion_matrix(y_true, y_pred, normalised=True, as_frame=False):

    """
    N x N matrix where N is number of distinct classes.
    Rows in this matrix represent actual values and columns the predicted ones.
    Therefore, entry :math:`a_{i, j}` represents a number of records
    whose actual class is `i` and their predicted class is `j`. 
    Thus, perfect model would have all entries within the matrix on its diagonal.

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_pred : iterale object
        Predicted values.
    normalised : bool, optional
        Normalize the metric by the total number of records.
    as_frame : bool, optional
        If you want to return it as a pandas dataframe.
    
    Returns
    -------
    confusion_matrix : :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        Confusion matrix as described above.

    Raises
    ------
    :class:`ValueError`
        If the inputted vectors are not of the same length.
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    check_consistent_length(y_true, y_pred)
    n = len(y_true)

    k = len(np.unique(y_true))
    label = {i: np.unique(y_true)[i] for i in range(k)}
    mapping = {np.unique(y_true)[i]: i for i in range(k)}

    # Initialise empty confusion matrix
    confusion_matrix = np.zeros((k, k)) # x: y_true, y: y_pred
    for i in range(n):
        t, p  = mapping[y_true[i]], mapping[y_pred[i]]
        confusion_matrix[t][p] += 1


    # Normalise by total number of testing instances (entire confusion matrix sums to 1)
    if normalised:
        confusion_matrix = confusion_matrix / n

    # convert output to pd.DataFrame for labeling Predicted/ Actual Class and Class Labels
    if as_frame:
        confusion_matrix = pd.DataFrame(confusion_matrix, label.values(), label.values())
        confusion_matrix = pd.concat([pd.concat([confusion_matrix], keys=['Predicted Class'], axis=1)],
                                                                    keys=['Actual Class'])

    return confusion_matrix

def precision_score(y_true, y_pred, average=None):

    """
    For each class `j`, return the ratio of the number
    of correct classificatin of `j` over number of times
    `j` was predicted in total.

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_pred : iterale object
        Predicted values.
    average : str, optional
        Aggregation metric for the precision scores of the respective classes.
    
    Returns
    -------
    precision_scores  or single score : :class:`numpy.ndarray` or float
        Numpy array where ith entry represents precision score for i-th class or single score if average is specified.
    """

    conf_matrix = confusion_matrix(y_true, y_pred, normalised=False)

    precision_scores = conf_matrix.diagonal() / np.sum(conf_matrix, axis=0)

    if average == 'macro':
        return np.mean(precision_scores)
    return precision_scores


def recall_score(y_true, y_pred, average=None):

    """
    For each class `j`, return the ratio of the number
    of correct classificatin of `j` over number of actual
    instances of `j`.

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_pred : iterale object
        Predicted values.
    average : str, optional
        Aggregation metric for the recall scores of the respective classes.
    
    Returns
    -------
    recall_scores  or single score : :class:`numpy.ndarray` or float
        Numpy array where ith entry represents recall score for i-th class or single score if average is specified.
    """
    conf_matrix = confusion_matrix(y_true, y_pred, normalised=False)

    recall_scores = conf_matrix.diagonal() / np.sum(conf_matrix, axis=1)

    if average == 'macro':
        return np.mean(recall_scores)
    return recall_scores

def f1_score(y_true, y_pred, average=None):

    """Harmonic mean of a precision and recall.

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_pred : iterale object
        Predicted values.
    average : str, optional
        Aggregation metric for the f1 scores of the respective classes.
    
    Returns
    -------
    f1_scores  or single score : 1d array or float
        Numpy array where ith entry represents f1 score for i-th class or single score if average is specified.
    """

    recall_scores = recall_score(y_true, y_pred, average=None)
    precision_scores = precision_score(y_true, y_pred, average=None)

    f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores)

    if average == 'macro':
        return np.mean(f1_scores)
 
    return f1_scores

def classification_report(y_true, y_pred):
    """
    Returns classification report which includes following info:

    - :func:`precision_score`
    - :func:`recall_score`
    - :func:`f1_score`
    - :func:`accuracy_score`
    - support = number of samples for each respective class

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_pred : iterale object
        Predicted values.

    Returns
    -------
    report : :class:`pandas.DataFrame`
        Report as described above.
    """

    k = len(np.unique(y_true))
    label = {i: np.unique(y_true)[i] for i in range(k)}
    mapping = {np.unique(y_true)[i]: i for i in range(k)}

    report = pd.DataFrame({'Precison': np.round(precision_score(y_true, y_pred), 2),
                           'Recall': np.round(recall_score(y_true, y_pred), 2),
                           'F1-Score': np.round(f1_score(y_true, y_pred), 2),
                           'Accuracy': ['' for _ in range(k -1)] + [np.round(accuracy_score(y_true, y_pred), 2)],
                           'Support': np.unique(y_true, return_counts=True)[1]}, label.values())

    return report
