import numpy as np
import tensorflow as tf
from .base_GAR import _GAR

def weights_from_storage(weights_loader, submissions):
    weights_list = []

    for submission in submissions:
        weights_cid = submission[3]
        weights = weights_loader.load(weights_cid)
        weights_list.append(weights)

    return weights_list
       

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def weighted_fed_avg(submissions, weights_loader):
  scaled_local_weight_list = []
  scaling_factor = 1/len(submissions)

  for i, submission in enumerate(submissions):
    weights_cid = submission[3]
    weights = weights_loader.load(weights_cid)
    scaled_weights = scale_model_weights(weights, scaling_factor)
    scaled_local_weight_list.append(scaled_weights)

  new_weights = sum_scaled_weights(scaled_local_weight_list)

  return new_weights

# courtesy of https://github.com/LPD-EPFL/AggregaThor
#  https://github.com/LPD-EPFL/AggregaThor/blob/master/aggregators/__init__.py
# 
class TFKrumGAR(_GAR):
  """ Full-TensorFlow Multi-Krum GAR class.
  """

  def __init__(self, nbworkers, nbbyzwrks):
    self.__nbworkers  = nbworkers
    self.__nbbyzwrks  = nbbyzwrks
    self.__nbselected = nbworkers - nbbyzwrks - 2

  def aggregate(self, gradients):
    with tf.name_scope("GAR_krum_tf"):
      # Assertion
      assert len(gradients) > 0, "Empty list of gradient to aggregate"
      # Distance computations
      distances = []
      for i in range(self.__nbworkers - 1):
        dists = list()
        for j in range(i + 1, self.__nbworkers):
            gr_i = []
            gr_j = []
            for w in gradients[i]:
              gr_i.extend(np.array(w).flatten())
            for w in gradients[j]:
              gr_j.extend(np.array(w).flatten())


            sqr_dst = tf.reduce_sum(tf.math.squared_difference(gr_i, gr_j))
            dists.append(tf.negative(tf.where(tf.math.is_finite(sqr_dst), sqr_dst, tf.constant(np.inf, dtype=sqr_dst.dtype)))) # Use of 'negative' to get the smallest distances and score indexes in 'nn.top_k'
        distances.append(dists)
      # Score computations
      scores = []
      for i in range(self.__nbworkers):
        dists = []
        for j in range(self.__nbworkers):
          if j == i:
            continue
          if j < i:
            dists.append(distances[j][i - j - 1])
          else:
            dists.append(distances[i][j - i - 1])
        dists = tf.parallel_stack(dists)
        dists, _ = tf.nn.top_k(dists, k=(self.__nbworkers - self.__nbbyzwrks - 2), sorted=False)
        scores.append(tf.reduce_sum(dists))
      # Average of the 'nbselected' smallest scoring gradients
      gradients = tf.parallel_stack(gradients)
      scores = tf.parallel_stack(scores)
      _, indexes = tf.nn.top_k(scores, k=self.__nbselected, sorted=False)
      return tf.reduce_mean(tf.gather(gradients, indexes), axis=0)
