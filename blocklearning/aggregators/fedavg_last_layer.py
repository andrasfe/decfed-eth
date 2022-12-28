import numpy as np

from blocklearning.models.simple_model import SimpleMLP
from .utils import *

class FedAvgOutputAggregator():
    def __init__(self, weights_loader, model_size = 10, image_lib = 'cifar'):
        self.weights_loader = weights_loader
        self.image_lib = image_lib
        self.model_size = model_size
        self.last_dense_index = -1 if image_lib == 'cifar' else -2
        self.model = SimpleMLP.build(self.image_lib)
    
    
    def __get_last_dense_layer_weights(self, model):
        return model.layers[self.last_dense_index].get_weights()[1]

    def __set_last_dense_layer_weights(self, model, last_dense_output):
        last_dense_weights = model.layers[self.last_dense_index].get_weights()
        last_dense_weights[1] = last_dense_output
        model.layers[self.last_dense_index].set_weights(last_dense_weights)

    def aggregate(self, submissions):
        
        agg_out = [0]*self.model_size

        for submission in submissions:
            weights_cid = submission[3]
            weights = self.weights_loader.load(weights_cid)
            self.model.set_weights(weights)
            out_layer = self.__get_last_dense_layer_weights(self.model)

            agg_out += out_layer

        agg_out = agg_out/self.model_size
        self.__set_last_dense_layer_weights(self.model, agg_out)
        return self.model
        