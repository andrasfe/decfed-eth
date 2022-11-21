from abc import ABC, abstractmethod

class BaseAlgo(ABC):
    def __init__(self, model, epochs, verbose):
        self.model = model
        self.epochs = epochs
        self.verbose = verbose
        

    @abstractmethod
    def fit(self, batched_ds):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    @abstractmethod
    def get_weights(self):
        pass

