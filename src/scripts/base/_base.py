from ..utils import ModelNotFittedError

class BaseModel:
    """
    BaseModel() that implements generic attributes and methods for each ML model.

    Warning: Don't use as a classifier or regressor. This class merely serves as a
    superclass from which other models inherit for the purpose of code reusability.
    """
    def __init__(self):
        self.X = self.y = self.n = self.p = None
        self.fitted = False

        self._model_name = self.__class__.__name__

    def is_fitted(self):
        return self.fitted

    def number_of_training_samples(self):
        if self.is_fitted():
            return self.n
        raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')

    def number_of_features(self):
        if self.is_fitted():
            return self.p
        raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')

    def __len__(self):
        return self.number_of_training_samples()

    def __repr__(self):
        return self._model_name

    """
    # for some reason overwrites the DecisionTree .fit() method, look into, solution
    # possibly using super()
    def fit(self, X, y):
        raise NotImplementedError(f'{self} does not implement .fit(X, y) yet')

    def predict(self, X):
        raise NotImplementedError(f'{self} does not implement .predict(X) yet')
    """
