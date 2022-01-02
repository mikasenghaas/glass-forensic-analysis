"""
The module includes basic plotting functionality for machine learning applications.
Namely, following functions are implemented:

- :func:`plot_1d_decision_regions`
- :func:`plot_2d_decision_regions`
"""

from ._classification import plot_1d_decision_regions
from ._classification import plot_2d_decision_regions

__all__ = [
        'plot_1d_decision_regions',
        'plot_2d_decision_regions'
        ]
