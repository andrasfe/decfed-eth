import json
import time
from .utilities import float_to_int
from .training_algos import RegularAlgo

class Trainer():
  def __init__(self, contract, weights_loader, model, data, logger = None, priv = None):
    self.logger = logger
    self.priv = priv
    self.weights_loader = weights_loader
    self.contract = contract
    self.train_ds_batched = data
    self.training_algo = RegularAlgo(model, 5, True)
    self.__register()

  def __log_info(self, message):
    if self.logger is not None:
      self.logger.info(message)


  def train(self):
    (round, weights_id) = self.contract.get_training_round()

    self.__log_info(json.dumps({ 'event': 'start', 'round': round, 'weights': weights_id, 'ts': time.time_ns() }))

    if weights_id != '':
      weights = self.weights_loader.load(weights_id)
      self.training_algo.set_weights(weights)

    self.__log_info(json.dumps({ 'event': 'train_start', 'round': round,'ts': time.time_ns() }))

    history = self.training_algo.fit(self.train_ds_batched)

    self.__log_info(json.dumps({ 'event': 'train_end', 'round': round,'ts': time.time_ns() }))

    trainingAccuracy = float_to_int(history.history["accuracy"][-1]*100)
    validationAccuracy = float_to_int(history.history["accuracy"][-1]*100)

    weights = self.training_algo.get_weights()
    if self.priv is not None:
      weights = self.priv.privatize(weights, validationAccuracy)

    weights_id = self.weights_loader.store(weights)

    submission = {
      'trainingAccuracy': trainingAccuracy,
      'testingAccuracy': validationAccuracy,
      'trainingDataPoints': 100,
      'weights': weights_id
    }
    self.contract.submit_submission(submission)

    self.__log_info(json.dumps({ 'event': 'end', 'round': round, 'weights': weights_id, 'ts': time.time_ns(), 'submission': submission }))

  # Private utilities
  def __register(self):
    self.__log_info(json.dumps({ 'event': 'checking_registration', 'ts': time.time_ns() }))

    self.contract.register_as_trainer()

    self.__log_info(json.dumps({ 'event': 'registration_checked', 'ts': time.time_ns() }))
