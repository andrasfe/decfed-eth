from tensorflow.keras.optimizers import SGD
from .base_algo import BaseAlgo

class BasilAlgo(BaseAlgo):

    def __init__(self, model, epochs, verbose):
        super().__init__(model, epochs, verbose)
        lr = 0.01 
        loss='categorical_crossentropy'
        metrics = ['accuracy']
        optimizer = SGD(lr=lr, 
                        decay=lr / self.epochs, 
                        momentum=0.9
                        )          

        self.model.compile(loss=loss, 
                optimizer=optimizer, 
                metrics=metrics)

    def fit(self, batched_ds):
        return self.model.fit(batched_ds, epochs=self.epochs, verbose=self.verbose)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
