from .utils import *

class FedAvgAggregator():
  def __init__(self, weights_loader):
    self.weights_loader = weights_loader

  def aggregate(self, submissions):
    return weighted_fed_avg(submissions, self.weights_loader)
