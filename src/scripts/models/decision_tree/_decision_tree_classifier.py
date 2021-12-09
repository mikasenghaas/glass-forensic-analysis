import math
from collections import Counter
import numpy as np 

from ...base import BaseClassifier
from ._decision_tree import DecisionTree
from ...metrics import binary_gini, gini, entropy

class DecisionTreeClassifier(BaseClassifier, DecisionTree):
    def __init__(self,
            criterion='gini',
            algorithm='greedy',
            max_depth=None,
            max_features='auto',
            min_samples_split=2,
            # max_nods = None,
            max_leaf_nodes=None,
            random_state=None):

        BaseClassifier.__init__(self)
        DecisionTree.__init__(
                self, 
                algorithm=algorithm, 
                max_depth=max_depth, 
                max_features=max_features,
                min_samples_split=min_samples_split,
                #max_nodes=max_nodes,
                max_leaf_nodes=max_leaf_nodes,
                random_state=random_state)

        if criterion == 'gini':
            self.criterion = gini
        elif criterion == 'binary_gini':
            self.criterion = binary_gini
        elif criterion == 'entropy':
            self.criterion = entropy
        else: 
            raise Exception('Cannot find this criterion')

    def _evaluate_leaf(self, node):
        labels = self.y[node.values]
        counter = Counter(labels)
        predict = counter.most_common()[0][0] # most_frequent class

        predict_proba = [0 for _ in range(self.k)]
        for pred, c in counter.items():
            predict_proba[self.intcode[pred]] = c / sum(counter.values())

        return predict, predict_proba
