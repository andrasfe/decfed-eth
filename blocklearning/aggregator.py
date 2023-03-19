import time
import json
from .training_algos import RegularAlgo

class Aggregator():
  def __init__(self, contract, weights_loader, model, train_ds, test_ds, aggregator, logger = None):
    self.logger = logger
    self.weights_loader = weights_loader
    self.contract = contract
    self.aggregator = aggregator
    self.model = model
    self.train_ds_batched = train_ds
    self.test_ds_batched = test_ds
    self.training_algo = RegularAlgo(model)
    self.__register()

  def __log_info(self, message):
    if self.logger is not None:  
      self.logger.info(message)


  def aggregate(self):
    (round, trainers, submissions) = self.contract.get_submissions_for_aggregation()

    self.__log_info(json.dumps({ 'event': 'start', 'submissions': submissions, 'round': round, 'ts': time.time_ns() }))

    new_weights = self.aggregator.aggregate(trainers, submissions)


    self.training_algo.set_weights(new_weights)
    acc, loss = self.training_algo.test(self.test_ds_batched)

    self.__log_info(json.dumps({ 'event': 'fedavg_end', 'round': round,'ts': time.time_ns(), 'accuracy': acc }))

    weights_cid = self.weights_loader.store(new_weights)
    self.contract.submit_aggregation(weights_cid)

    self.__log_info(json.dumps({ 'event': 'end', 'round': round, 'submissions': submissions, 'ts': time.time_ns(), 'new_weights': weights_cid }))

  # Private utilities
  def __register(self):
    self.__log_info(json.dumps({ 'event': 'checking_registration', 'ts': time.time_ns() }))

    self.contract.register_as_aggregator()

    self.__log_info(json.dumps({ 'event': 'registration_checked', 'ts': time.time_ns() }))
