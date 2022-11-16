import numpy as np
import tensorflow as tf

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

def weighted_fed_avg(submissions, model_size, weights_loader, avg_weights):
  scaled_local_weight_list = []
  scaling_factor = 1/len(submissions)

  for i, submission in enumerate(submissions):
    weights_cid = submission['weights']
    weights = weights_loader.load(weights_cid)
    scaled_weights = scale_model_weights(weights, scaling_factor)
    scaled_local_weight_list.append(scaled_weights)

  new_weights = sum_scaled_weights(scaled_local_weight_list)

  return new_weights
