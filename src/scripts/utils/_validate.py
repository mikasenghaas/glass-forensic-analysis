import numpy as np

def check_consistent_length(*arr):

    """Checks whether the provided set of nd-arrays have the same first dimension.

    Raises
    ------
    ValueError
        If the len of input arrays is incosistent.
    """

    lengths = [X.shape[0] for X in arr if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples:"\
           f"{[int(l) for l in lengths]}"
        )

def validate_feature_matrix(X):
    """
    Makes any inserted matrix with less than 3 dimensions into a 2d-feature matrix.

    Parameters
    ----------
    X : :class:`numpy.ndarray`

    Returns
    -------
    X : :class:`numpy.ndarray`
        2d array.

    Raises
    ------
    AssetionError
        If X is 3 dimensional.
    """

    X = np.array(X)

    assert len(X.shape) <= 2, 'Cannot work with 3d-feature matrix'

    if len(X.shape)<2:
        return X.reshape(-1, 1)
    else:
        return X

def validate_target_vector(y):
    """Makes sure that the target vector is one dimensional.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Target vector.
    
    Returns
    -------
    y : :class:`numpy.ndarray`
        1d target vector.
    """
    y = np.array(y)

    assert len(y.shape) == 1, 'Target Vector must be 1d'

    return y
