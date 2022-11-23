import json
import time
from .base_trainer import BaseTrainer
from ..utilities import float_to_int
from ..training_algos import RegularAlgo

class PeerAggregatingTrainer(BaseTrainer):
  def __init__(self, contract, weights_loader, model, train_data, test_data, aggregator, logger = None, priv = None):
    self.logger = logger
    self.priv = priv
    self.weights_loader = weights_loader
    self.contract = contract
    self.train_ds_batched = train_data
    self.test_ds_data = test_data
    self.training_algo = RegularAlgo(model, 5, True)
    self.aggregator = aggregator
    super().__init__()

  def train(self):
    (round, weights_id) = self.contract.get_training_round()

    weights = None
    # aggregate other trainers models before training own
    if round > 1:
        (_, trainers, submissions) = self.contract.get_submissions_from_prior_round()

        self._log_info(json.dumps({ 'event': 'self_agg_start', 'round': round, 'ts': time.time_ns() }))

        weights = self.aggregator.aggregate(trainers, submissions, None, None)

        self._log_info(json.dumps({ 'event': 'self_agg_end', 'round': round,'ts': time.time_ns() }))


    self._log_info(json.dumps({ 'event': 'start', 'round': round, 'weights': weights_id, 'ts': time.time_ns() }))

    if weights == None and weights_id != '':
      weights = self.weights_loader.load(weights_id)
      self.training_algo.set_weights(weights)

    self._log_info(json.dumps({ 'event': 'train_start', 'round': round,'ts': time.time_ns() }))

    history = self.training_algo.fit(self.train_ds_batched)

    self._log_info(json.dumps({ 'event': 'train_end', 'round': round,'ts': time.time_ns() }))

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

    self._log_info(json.dumps({ 'event': 'end', 'round': round, 'weights': weights_id, 'ts': time.time_ns(), 'submission': submission }))
