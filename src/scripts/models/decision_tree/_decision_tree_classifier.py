import math
import numpy as np

from ...base import BaseClassifier
from ._decision_tree import DecisionTree

class DecisionTreeClassifier(BaseClassifier, DecisionTree):

    """Class allowing to specify parameters of the model.

    This class serves as a high level client facing API.
    It allows to specify all hyper parameters important 
    for training process.

    Parameters
    ----------
    criterion : str, optional
        How to calculate impurity of nodes.
    
    algorithm : str, optional
        How to approach building process of the tree.
    
    max_depth : int, optional
        Maximum depth of the tree.

    max_features : int or float or str, optional
        How many features to consider during each split of the node.
    
    min_samples_split : int, optional
        Minimum samples required to be present within the node in order
        to be able to further split it.
    
    random_state : int, optional
        If using `random` alforithm, it is useful to specify this parameter
        in order to ensure reproducibility of results.
    """

    def __init__(self,
            algorithm='greedy',
            criterion='gini',
            max_depth=None,
            max_features='auto',
            min_samples_split=2,
            random_state=None):

        BaseClassifier.__init__(self)

        DecisionTree.__init__(
                self, 
                algorithm=algorithm,
                criterion=criterion,
                max_depth=max_depth, 
                max_features=max_features,
                min_samples_split=min_samples_split,
                random_state=random_state)
