import numpy as np
from .utils import *

class MultiKrumAggregator():
  def __init__(self, weights_loader, model_size):
    self.model_size = model_size
    self.weights_loader = weights_loader

  def aggregate(self, trainers, submissions):
    weights_cids = [cid for (_, _, _, cid, _, _) in submissions]
    weights = [self.weights_loader.load(cid) for cid in weights_cids]

    return multikrum_aggregate(weights, trainers)

