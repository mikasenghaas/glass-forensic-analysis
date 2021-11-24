import numpy as np

def check_consistent_length(*arr):
    lengths = [X.shape[0] for X in arr if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples:"\
           f"{[int(l) for l in lengths]}"
        )

def validate_feature_matrix(X):
    # makes any inserted matrix with less than 3 dimensions into a 2d-feature matrix
    X = np.array(X)

    assert len(X.shape) <= 2, 'Cannot work with 3d-feature matrix'

    if len(X.shape)<2:
        return X.reshape(-1, 1)
    else:
        return X

def validate_target_vector(y):
    y = np.array(y)

    assert len(y.shape) == 1, 'Target Vector must be 1d'

    return y
