from ..utils import ModelNotFittedError

class BaseModel:
    """Implements generic attributes and methods for each ML model.

    Warnings
    --------
    Don't use as a classifier or regressor. This class merely serves as a
    superclass from which other models inherit for the purpose of code reusability.
    """
    def __init__(self):
        self.X = self.y = self.n = self.p = None
        self.fitted = False
        self._model_name = self.__class__.__name__

    def is_fitted(self):
        """Is given model fitted?

        Returns
        -------
        bool
        """
        return self.fitted

    def number_of_training_samples(self):
        """Return number provided training samples.

        Returns
        -------
        int
            number provided training samples

        Raises
        ------
        :class:`ModelNotFittedError`
            If the model has not been fitter yet.
        """
        if self.is_fitted():
            return self.n
        raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')

    def number_of_features(self):
        """Returns number of features within given training data-set.

        Returns
        -------
        int
            Number of features.
        Raises
        ------
        :class:`ModelNotFittedError`
            If the model has not been fitter yet.
        """

        if self.is_fitted():
            return self.p
        raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')

    def __len__(self):
        return self.number_of_training_samples()

    def __repr__(self):
        return self._model_name
