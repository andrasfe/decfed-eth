import numpy as np
import tensorflow as tf

def score(weights, trainers):
    R = len(weights)
    f = R // 3 - 1
    closest_updates = R - f - 2

    scores = []

    for i in range(len(weights)):
      dists = []

      for j in range(len(weights)):
        if i == j:
          continue

        diff = np.subtract(weights[j],weights[i])
        l2_norm = np.sqrt(np.sum([np.sum(np.square(w)) for w in diff]))
        dists.append(l2_norm)

      dists_sorted = np.argsort(dists)[:closest_updates]
      score = np.array([dists[i] for i in dists_sorted]).sum()
      scores.append(score)
    return trainers, scores



def local_score(weights, my_weights, other_trainers):
    R = len(weights)
    f = R // 3 - 1
    closest_updates = R - f - 2

    dists = []

    for i in range(len(weights)):
        diff = np.subtract(weights[i], my_weights)
        l2_norm = np.sqrt(np.sum([np.sum(np.square(w)) for w in diff]))
        dists.append(l2_norm)

    dists_sorted = np.argsort(dists)[:closest_updates]
    return [other_trainers[i] for i in dists_sorted]

def fed_avg(weights):
    assert(len(weights) > 0)
    n_layers = len(weights[0])

    avg_weights = list()

    for layer in range(n_layers):
        layer_weights = np.array([w[layer] for w in weights])
        mean_layer_weights = np.mean(layer_weights, axis = 0)
        avg_weights.append(mean_layer_weights)

    return avg_weights


