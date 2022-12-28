
from tensorflow.keras.optimizers import SGD
from .base_algo import BaseAlgo
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

class RegularAlgo(BaseAlgo):

    def __init__(self, model, epochs=1, image_lib = 'cifar', verbose=False):
        super().__init__(model, epochs, verbose)
        lr = 0.01 
        loss='categorical_crossentropy'
        metrics = ['accuracy']
        optimizer = SGD(learning_rate=lr, 
                        decay=lr / self.epochs, 
                        momentum=0.9
                        )          
        self.model.compile(loss=loss, 
                optimizer=optimizer, 
                metrics=metrics)
        self.image_lib = image_lib

    def __freeze_except_last(self):
        idx = 2 if self.image_lib == 'cifar' else 3
        for layer in self.model.layers[:-idx]:
            layer.trainable=False
        

    def fit(self, batched_ds, freeze_except_last = False):
        if  freeze_except_last:
            self.__freeze_except_last()

        return self.model.fit(batched_ds, epochs=self.epochs, verbose=self.verbose)

    def test(self, batched_ds):
        acc = []
        loss = []
        for slice in batched_ds:
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            logits = self.model.predict(slice[0])
            loss.append(cce(slice[1], logits).numpy())
            acc.append(accuracy_score(tf.argmax(logits, axis=1), tf.argmax(slice[1], axis=1)))
        return sum(acc)/len(acc), sum(loss)/len(loss)

    def set_model(self, model):
        self.model = model

    def set_weights(self, weights, freeze_except_last = False):
        self.model.set_weights(weights)

        if  freeze_except_last:
            self.__freeze_except_last()

    def get_weights(self):
        return self.model.get_weights()

    def get_model(self):
        return self.model


