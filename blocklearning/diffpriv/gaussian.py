import numpy as np
from diffprivlib.mechanisms import GaussianAnalytic


class Gaussian():
  def __init__(self, epsilon = 1e+6, delta = 1.0e-4, sensitivity = 1):
    self.mech = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

  def __get_last_dense_layer_output_weights(self, model):
    return model.layers[-2].get_weights()

  def privatized_weights(self, model):
    x = self.__get_last_dense_layer_output_weights(model)
    y = x[1]

    for i in range(len(y)):
      y[i] = self.mech.randomise(y[i])

    model.layers[-2].set_weights(x)
    return model.get_weights()