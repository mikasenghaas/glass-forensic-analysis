"""
This module serves as a support for other modules.
The following helper functions are defined:

- :func:`check_consistent_length`
- :func:`validate_feature_matrix`
- :func:`validate_target_vector`
- :func:`get_data`
- :func:`generate_summary`
- :func:`ModelNotFittedError`

"""

from ._validate import check_consistent_length
from ._validate import validate_feature_matrix
from ._validate import validate_target_vector

from ._exceptions import ModelNotFittedError
from ._colors import COLORS
from ._loader import get_data
from ._generate_summary import generate_summary

__all__ = [
        'check_consistent_length',
        'validate_feature_matrix',
        'validate_target_vector',
        'get_data',
        'generate_summary',
        'ModelNotFittedError',
        'COLORS'
]
