from ._base import BaseModel

from ..utils import ModelNotFittedError
from ..metrics import accuracy_score

class BaseClassifier(BaseModel):
    def __init__(self):
        super().__init__() 
        self.k = None
        self.label = {}
        self.intcode = {}

    def classes(self):
        if not self.is_fitted():
            raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')
        
        return self.label.values()

    def number_of_classes(self):
        if not self.is_fitted():
            raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')
        
        return self.k 

    def score(self):
        if not self.is_fitted():
            raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')

        training_preds = self.predict(self.X)
        return accuracy_score(self.y, training_preds)
