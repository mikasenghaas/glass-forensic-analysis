"""
Submodule :mod:eduml.utils that contains useful functionalities/ helper functions
when dealing with machine learning algorithms
"""

from ._validate import check_consistent_length
from ._validate import validate_feature_matrix
from ._validate import validate_target_vector

from ._colors import COLORS


__all__ = [
        'check_consistent_length',
        'validate_feature_matrix',
        'validate_target_vector',

        'COLORS' 
        ]
