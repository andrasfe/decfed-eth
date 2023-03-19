import numpy as np
from .utils import *

class MultiKrumAggregator():
  def __init__(self, weights_loader, model_size):
    self.model_size = model_size
    self.weights_loader = weights_loader

  def __multikrum_aggregate(self, weights, trainers):
      assert(len(weights) == len(trainers))
      _, _, selected_weights = self.__multikrum_selected_trainers(weights, trainers)
      return fed_avg(selected_weights)

  def __multikrum_selected_trainers(self, weights, trainers):    
      trainers, scores = score(weights, trainers)

      medians = []

      for t, _ in enumerate(trainers):
        medians.append(np.median(scores[t]))

      R = len(weights)
      f = R // 3 - 1

      sorted_idxs = np.argsort(medians)
      lowest_idxs = sorted_idxs[:R-f]
      
      honest_trainers = []
      dishonest_trainers = []
      
      for i in range(len(trainers)):
          if i in lowest_idxs:
            honest_trainers.append(trainers[i])
          else:
            dishonest_trainers.append(trainers[i]) 

      return honest_trainers, dishonest_trainers, [weights[i] for i in lowest_idxs]    

  def aggregate(self, trainers, submissions):
    weights_cids = [cid for (_, _, _, cid, _, _) in submissions]
    weights = [self.weights_loader.load(cid) for cid in weights_cids]

    return self.__multikrum_aggregate(weights, trainers)
  
  def assess_trainers(self, trainers, submissions):
    weights_cids = [cid for (_, _, _, cid, _, _) in submissions]
    weights = [self.weights_loader.load(cid) for cid in weights_cids]

    return self.__multikrum_selected_trainers(weights, trainers)

