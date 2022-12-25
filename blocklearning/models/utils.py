import numpy as np

def get_layer_info(model):
    layers = []
    total = 0
    for layer in model.get_weights():
        shape = layer.shape
        weights = np.prod(shape)
        total += weights
        layers.append((shape, weights))
    return layers, total