from .utils import *

class FedAvgAggregator():
  def __init__(self, model_size, weights_loader):
    self.model_size = model_size
    self.weights_loader = weights_loader

  def aggregate(self, trainers, submissions, scorers = None, scores = None):
    return weighted_fed_avg(submissions, self.weights_loader)
