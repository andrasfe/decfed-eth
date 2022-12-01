import json
import time
from .base_trainer import BaseTrainer
from ..utilities import float_to_int
from ..training_algos import RegularAlgo
from blocklearning.contract import RoundPhase
from random import randint

class PeerAggregatingTrainer(BaseTrainer):
  def __init__(self, contract, pedersen, weights_loader, model, train_data, test_data, aggregator, logger = None, priv = None, rounds=3):
    self.logger = logger
    self.priv = priv
    self.weights_loader = weights_loader
    self.contract = contract
    self.pedersen = pedersen
    self.train_ds_batched = train_data
    self.test_ds_batched = test_data
    self.training_algo = RegularAlgo(model, rounds, True)
    self.aggregator = aggregator
    self.__hiddenWeights = ''
    self.__random_T = 0
    self.commitment = None
    super().__init__()

  def __do_first_update(self):
    history = self.training_algo.fit(self.train_ds_batched)
    acc, loss = self.training_algo.test(self.test_ds_batched)
    trainingAccuracy = float_to_int(history.history["accuracy"][-1]*100)
    validationAccuracy = float_to_int(acc*100)
    weights = self.training_algo.get_weights()
    self.__random_T = randint(0, 1e+8)
    # load trained model to IPFS and commit to address
    self.__hiddenWeights = self.weights_loader.store(weights)
    self.commitment = self.pedersen.get_commitment(self.__random_T, self.__hiddenWeights)
    submission = {
      'trainingAccuracy': trainingAccuracy,
      'testingAccuracy': validationAccuracy,
      'trainingDataPoints': 100,
      'weights': '',
      'firstCommit': self.commitment[0],
      'secondCommit': self.commitment[1]
    }

    self.contract.submit_first_update(submission)    

  def __do_proof_presentment(self):
    self.contract.validate_pedersen(self.__r, self.__hiddenWeights)


  def train(self):
    (round, weights_id) = self.contract.get_training_round()

    if weights_id != '':
      weights = self.weights_loader.load(weights_id)
      self.training_algo.set_weights(weights, freeze_except_last_dense=True)


    phase = self.contract.get_round_phase()
        
    if phase == RoundPhase.WAITING_FOR_FIRST_UPDATE:
      self.__do_first_update()
    elif phase == RoundPhase.WAITING_FOR_PROOF_PRESENTMENT:
      self.__do_proof_presentment()
    elif phase == RoundPhase.WAITING_FOR_UPDATES:
      (_, trainers, submissions) = self.contract.get_submissions_from_prior_round()
      self._log_info(json.dumps({ 'event': 'self_agg_start', 'round': round, 'ts': time.time_ns() }))

      history = self.training_algo.fit(self.train_ds_batched)
      new_model = self.aggregator.aggregate(self.training_algo.get_model(), submissions, self.test_ds_batched)
      self.training_algo.set_model(new_model)
      acc, loss = self.training_algo.test(self.test_ds_batched)
      self._log_info(json.dumps({ 'event': 'self_agg_end', 'round': round,'ts': time.time_ns() }))

      trainingAccuracy = float_to_int(history.history["accuracy"][-1]*100)
      validationAccuracy = float_to_int(acc*100)

      weights = self.training_algo.get_weights()
      if self.priv is not None:
        weights = self.priv.privatize(weights, validationAccuracy)

      weights_id = self.weights_loader.store(weights)

      submission = {
        'trainingAccuracy': trainingAccuracy,
        'testingAccuracy': validationAccuracy,
        'trainingDataPoints': 100,
        'weights': weights_id,
        'firstCommit': 0,
        'secondCommit': 0
      }
      self.contract.submit_submission(submission)

      self._log_info(json.dumps({ 'event': 'end', 'round': round, 'weights': weights_id, 'ts': time.time_ns(), 'submission': submission }))

