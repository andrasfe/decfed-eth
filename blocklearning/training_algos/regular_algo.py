
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy import DPKerasSGDOptimizer
from .base_algo import BaseAlgo
from sklearn.metrics import accuracy_score

class RegularAlgo(BaseAlgo):

    def __init__(self, model, epochs=1, image_lib='mnist', batch_size=30, verbose=False):
        super().__init__(model, epochs, verbose)
        l2_norm_clip = 1.5
        noise_multiplier = 1.3
        num_microbatches = 30
        learning_rate = 0.25

        if batch_size % num_microbatches:
            raise ValueError('Batch size should be an integer multiple of the number of microbatches')

        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)
        
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.losses.Reduction.NONE)

        self.model.compile(optimizer=optimizer, 
                           loss=loss, 
                           metrics=['accuracy'])

    def __freeze_except_last(self):
        idx = 2 if self.image_lib == 'cifar' else 3
        stop_index = -idx
        for layer in self.model.layers[:stop_index]:
            layer.trainable = False
        

    def fit(self, batched_ds, freeze_except_last=False):
        if freeze_except_last:
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

    def set_weights(self, weights, freeze_except_last=False):
        self.model.set_weights(weights)

        if freeze_except_last:
            self.__freeze_except_last()

    def get_weights(self):
        return self.model.get_weights()

    def get_model(self):
        return self.model


