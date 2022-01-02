"""
Base module includes three main classes:

- :class:`BaseModel`
- :class:`BaseClassifier`

These classes implement attributes and methods which are common to all
models and classifiers.
"""

from ._base import BaseModel
from ._base_classifier import BaseClassifier

__all__ = [
        'BaseModel',
        'BaseClassifier',
        ]
