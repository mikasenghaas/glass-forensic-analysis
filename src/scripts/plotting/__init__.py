"""
The :mod:`eduml.plotting` module includes basic plotting functionality  
for machine learning applications.
"""

from ._classification import plot_1d_decision_regions
from ._classification import plot_2d_decision_regions
from ._classification import plot_decision_regions

from ._regression import plot_1d_regression

__all__ = [
        'plot_decision_regions'
        ]
