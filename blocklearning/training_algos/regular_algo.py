
from tensorflow.keras.optimizers import SGD
from .base_algo import BaseAlgo
import tensorflow as tf
from sklearn.metrics import accuracy_score

class RegularAlgo(BaseAlgo):

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

    def test(self, batched_ds):
        acc = []
        loss = []
        for slice in batched_ds:
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            logits = self.model.predict(slice[0])
            loss.append(cce(slice[1], logits))
            acc.append(accuracy_score(tf.argmax(logits, axis=1), tf.argmax(slice[1], axis=1)))
        return sum(acc)/len(acc), sum(loss)/len(loss)

    def set_model(self, model):
        self.model = model

    def set_weights(self, weights, freeze_except_last_dense = False):
        self.model.set_weights(weights)

        if  freeze_except_last_dense:
            for layer in self.model.layers[:-2]:
                layer.trainable=False

    def get_weights(self):
        return self.model.get_weights()

    def get_model(self):
        return self.model


