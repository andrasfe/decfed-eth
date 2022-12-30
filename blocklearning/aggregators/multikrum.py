import numpy as np
from .utils import *

class MultiKrumAggregator():
  def __init__(self, weights_loader, model_size):
    self.model_size = model_size
    self.weights_loader = weights_loader

  def aggregate(self, trainers, submissions):
    trainers, scores, weights = score(self.weights_loader, trainers, submissions)
    medians = []
    for t, trainer in enumerate(trainers):
      medians.append(np.median(scores[t]))
    print('medians', medians)

    R = len(submissions)
    f = R // 3 - 1

    sorted_idxs = np.argsort(medians)
    lowest_idxs = sorted_idxs[:R-f]
    selected_weights = [weights[i] for i in lowest_idxs]

    return fed_avg(selected_weights)

