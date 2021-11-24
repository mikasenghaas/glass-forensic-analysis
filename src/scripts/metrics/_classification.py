"""
The :file:`eduml.metrics._classification` module includes score functions, performance 
metrics for classification problems.
"""

import numpy as np
import pandas as pd
from ..utils import check_consistent_length

def accuracy_score(y_true, y_pred, normalised=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    check_consistent_length(y_true, y_pred)
    n = len(y_true)

    correct = sum(y_true == y_pred)

    if normalised:
        return correct / n
    return correct

def classification_error(y_true, y_pred, normalised=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    check_consistent_length(y_true, y_pred)
    n = len(y_true)

    misclassified = sum(y_true != y_pred)

    if normalised:
        return misclassified / n 
    return misclassified

def confusion_matrix(y_true, y_pred, normalised=True, as_frame=False):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    check_consistent_length(y_true, y_pred)
    n = len(y_true)

    k = len(np.unique(y_true))
    label = {i: np.unique(y_true)[i] for i in range(k)}
    mapping = {np.unique(y_true)[i]: i for i in range(k)}

    # initialise empty confusion matrix
    confusion_matrix = np.zeros((k, k)) # x: y_true, y: y_pred
    for i in range(n):
        t, p  = mapping[y_true[i]], mapping[y_pred[i]]
        confusion_matrix[t][p] += 1


    # normalise by total number of testing instances (entire confusion matrix sums to 1)
    if normalised:
        confusion_matrix = confusion_matrix / n

    # convert output to pd.DataFrame for labeling Predicted/ Actual Class and Class Labels
    if as_frame:
        confusion_matrix = pd.DataFrame(confusion_matrix, label.values(), label.values())
        confusion_matrix = pd.concat([pd.concat([confusion_matrix], keys=['Predicted Class'], axis=1)],
                                                                    keys=['Actual Class'])

    return confusion_matrix

def precision_score(y_true, y_pred, average=None):
    conf_matrix = confusion_matrix(y_true, y_pred, normalised=False)

    precision_scores = conf_matrix.diagonal() / np.sum(conf_matrix, axis=0)

    if average == 'macro':
        return np.mean(precision_scores)
    return precision_scores


def recall_score(y_true, y_pred, average=None):
    conf_matrix = confusion_matrix(y_true, y_pred, normalised=False)

    recall_scores = conf_matrix.diagonal() / np.sum(conf_matrix, axis=1)

    if average == 'macro':
        return np.mean(recall_scores)
    return recall_scores

def f1_score(y_true, y_pred, average=None):
    recall_scores = recall_score(y_true, y_pred, average=None)
    precision_scores = precision_score(y_true, y_pred, average=None)

    f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores)

    if average == 'macro':
        return np.mean(f1_scores)
 
    return f1_scores

def classification_report(y_true, y_pred):
    k = len(np.unique(y_true))
    label = {i: np.unique(y_true)[i] for i in range(k)}
    mapping = {np.unique(y_true)[i]: i for i in range(k)}

    report = pd.DataFrame({'Precison': np.round(precision_score(y_true, y_pred), 2),
                           'Recall': np.round(recall_score(y_true, y_pred), 2),
                           'F1-Score': np.round(f1_score(y_true, y_pred), 2),
                           'Accuracy': ['' for _ in range(k -1)] + [np.round(accuracy_score(y_true, y_pred), 2)],
                           'Support': np.unique(y_true, return_counts=True)[1]}, label.values())

    return report




