from ._base import BaseModel
from ..utils import ModelNotFittedError
from ..metrics import accuracy_score

class BaseClassifier(BaseModel):

    """Implement methods and attributes common for all classifiers.
    """

    def __init__(self):
        super().__init__() 
        self.k = None
        self.label = {}
        self.intcode = {}

    def classes(self):
        """Return unique classes from the given training dataset.

        Returns
        -------
        Iterable
            Python iterable with all the unique classses.
        
        Raises
        ------
        :class:`ModelNotFittedError`
            If the model has not been fitter yet.
        """

        if not self.is_fitted():
            raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')
        
        return self.label.values()

    def number_of_classes(self):
        """
        Return number of unique classes based on the provided training dataset.

        Returns
        -------
        int
            Number of unique classses.
        
        Raises
        ------
        :class:`ModelNotFittedError`
            If the model has not been fitter yet.
        """

        if not self.is_fitted():
            raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')
        
        return self.k 

    def score(self):
        """Return training accuracy score.

        Returns
        -------
        float
            Training accuracy score.

        Raises
        ------
        :class:`ModelNotFittedError`
            If the model has not been fitter yet.
        """
        if not self.is_fitted():
            raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')

        training_preds = self.predict(self.X)
        return accuracy_score(self.y, training_preds)
