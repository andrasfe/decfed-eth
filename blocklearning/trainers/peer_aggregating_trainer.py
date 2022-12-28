import json
import time
from .base_trainer import BaseTrainer
from ..utilities import float_to_int
from ..training_algos import RegularAlgo
from blocklearning.contract import RoundPhase
from random import randint

class PeerAggregatingTrainer(BaseTrainer):
  def __init__(self, contract, pedersen, weights_loader, model, train_data, test_data, aggregator, logger = None, priv = None, rounds=5):
    self.logger = logger
    self.priv = priv
    self.weights_loader = weights_loader
    self.contract = contract
    self.pedersen = pedersen
    self.train_ds_batched = train_data
    self.test_ds_batched = test_data
    self.training_algo = RegularAlgo(model, rounds, False)
    self.aggregator = aggregator
    self.__hiddenWeights = ''
    self.__random_T = 0
    self.commitment = None
    super().__init__()

  def __load_weights_by_id(self, weights_id):
    if weights_id != '':
      weights = self.weights_loader.load(weights_id)
      self.training_algo.set_weights(weights, freeze_except_last=True)


  def __do_first_update(self):
    history = self.training_algo.fit(self.train_ds_batched, freeze_except_last=True)
    acc, loss = self.training_algo.test(self.test_ds_batched)
    trainingAccuracy = float_to_int(history.history["accuracy"][-1]*100)
    validationAccuracy = float_to_int(acc*100)

    weights = None
    
    if self.priv is None:
      weights = self.training_algo.get_weights()
    else:
      weights = self.priv.privatized_weights(self.training_algo.get_model())

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
    self.contract.validate_pedersen(self.__random_T, self.__hiddenWeights)


  def train(self):
    (round, weights_id) = self.contract.get_training_round()

    phase = self.contract.get_round_phase()
        
    if phase == RoundPhase.WAITING_FOR_FIRST_UPDATE:
      self.__load_weights_by_id(weights_id)
      self.__do_first_update()
    elif phase == RoundPhase.WAITING_FOR_PROOF_PRESENTMENT:
      self.__do_proof_presentment()
    elif phase == RoundPhase.WAITING_FOR_UPDATES:
      self.__load_weights_by_id(weights_id)
      (_, trainers, submissions) = self.contract.get_submissions_for_round(round - 1)
      self._log_info(json.dumps({ 'event': 'self_agg_start', 'round': round, 'ts': time.time_ns() }))

      my_index = trainers.index(self.contract.account)
      submissions.pop(my_index)    

      history = self.training_algo.fit(self.train_ds_batched)
      pre_acc, pre_loss = self.training_algo.test(self.test_ds_batched)
      new_model = self.aggregator.aggregate(self.training_algo.get_model(), submissions, self.test_ds_batched)
      self.training_algo.set_model(new_model)
      acc, loss = self.training_algo.test(self.test_ds_batched)
      self._log_info(json.dumps({ 'event': 'self_agg_post', 'round': round,'ts': time.time_ns(), 'pre_acc': pre_acc, 'acc': acc, 'pre_loss': pre_loss, 'loss': loss }))

      trainingAccuracy = float_to_int(history.history["accuracy"][-1]*100)
      validationAccuracy = float_to_int(acc*100)

      weights = None

      if self.priv is None:
        weights = self.training_algo.get_weights()
      else:
        weights = self.priv.privatized_weights(self.training_algo.get_model())

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

