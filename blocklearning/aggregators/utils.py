import numpy as np

# def score(weights, trainers):
#     R = len(weights)
#     f = R // 3 - 1
#     closest_updates = R - f - 2

#     scores = []

#     for i in range(len(weights)):
#       dists = []

#       for j in range(len(weights)):
#         if i == j:
#           continue

#         diff = np.subtract(weights[j],weights[i])
#         l2_norm = np.sqrt(np.sum([np.sum(np.square(w)) for w in diff]))
#         dists.append(l2_norm)

#       dists_sorted = np.argsort(dists)[:closest_updates]
#       score = np.array([dists[i] for i in dists_sorted]).sum()
#       scores.append(score)
#     return trainers, scores

import tensorflow as tf

def score(weights, trainers):
    R = len(weights)
    f = R // 3 - 1
    closest_updates = R - f - 2

    # Convert weights to TensorFlow tensors
    weights_tensors = [tf.convert_to_tensor(w, dtype=tf.float32) for w in weights]

    # Calculate pairwise L2 distances
    dists = tf.norm(tf.expand_dims(weights_tensors, 1) - weights_tensors, ord='euclidean', axis=2)

    # Set the diagonal to infinity to exclude self-distances
    dists = tf.linalg.set_diag(dists, tf.fill(dists.shape[0], tf.constant(float('inf'), dtype=tf.float32)))

    # Sort distances and find closest updates
    dists_sorted = tf.argsort(dists)
    closest_indices = dists_sorted[:, :closest_updates]

    # Calculate scores
    closest_dists = tf.gather(dists, closest_indices, axis=1)
    scores = tf.reduce_sum(closest_dists, axis=1)

    return trainers, scores.numpy().tolist()

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


