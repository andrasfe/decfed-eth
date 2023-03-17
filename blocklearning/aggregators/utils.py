import numpy as np
import tensorflow as tf
from ..utilities import floats_to_ints

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
    return trainers, scores, weights

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

def multikrum_aggregate(weights, trainers):
    assert(len(weights) == len(trainers))
    trainers, scores, weights = score(weights, trainers)

    medians = []

    for t, trainer in enumerate(trainers):
      medians.append(np.median(scores[t]))
    print('medians', medians)

    R = len(weights)
    f = R // 3 - 1

    sorted_idxs = np.argsort(medians)
    lowest_idxs = sorted_idxs[:R-f]
    selected_weights = [weights[i] for i in lowest_idxs]

    return fed_avg(selected_weights)
