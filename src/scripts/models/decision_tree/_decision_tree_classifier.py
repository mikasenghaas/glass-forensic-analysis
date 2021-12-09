import math
from collections import Counter
import numpy as np 

from ._decision_tree import DecisionTree
from ...metrics import binary_gini, gini, entropy

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, criterion='gini', max_depth=None, algorithm='greedy', max_features='auto'):

        super().__init__(max_depth=max_depth, algorithm=algorithm, max_features=max_features)

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
            predict_proba[pred] = c / sum(counter.values())

        return predict, predict_proba
