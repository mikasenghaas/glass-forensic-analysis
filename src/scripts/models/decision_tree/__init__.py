"""
Decision Tree module includes three core classes:

- :class:`DecisionTreeClassifier`,
- :class:`DecisionTree`
- :class:`Node`

These classes all together allow to train Decision Tree model on any dataset and then perform classification.
"""

from ._node import Node
from ._decision_tree import DecisionTree
from ._decision_tree_classifier import DecisionTreeClassifier

__all__ = [
        'Node',
        'DecisionTree',
        'DecisionTreeClassifier'
        ]
