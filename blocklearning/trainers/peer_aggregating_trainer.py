import json
import time
from .base_trainer import BaseTrainer
from blocklearning.utilities import float_to_int
from blocklearning.training_algos import RegularAlgo
from blocklearning.contract import RoundPhase
from blocklearning.aggregators import ModelSelector
from random import randint

class PeerAggregatingTrainer(BaseTrainer):
  def __init__(self, contract, pedersen, weights_loader, model, train_data, test_data, aggregator, logger = None, rounds=30):
    self.logger = logger
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
    self.last_action_completed = None
    super().__init__()

  def __load_weights_by_id(self, weights_id):
    if weights_id != '':
      weights = self.weights_loader.load(weights_id)
      self.training_algo.set_weights(weights, freeze_except_last=False)


  def __do_first_update(self, phase):
    history = self.training_algo.fit(self.train_ds_batched, freeze_except_last=False)
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
    self.last_action_completed = RoundPhase.WAITING_FOR_FIRST_UPDATE

  def __do_proof_presentment(self, phase):
    self.contract.validate_pedersen(self.__random_T, self.__hiddenWeights)
    self.last_action_completed = RoundPhase.WAITING_FOR_PROOF_PRESENTMENT

  def __do_updates(self, round, phase, weights_id):
    self.__load_weights_by_id(weights_id)

    acc, loss = self.training_algo.test(self.test_ds_batched)
    (_, trainers, submissions) = self.contract.get_submissions_for_round(round - 1)

    my_index = trainers.index(self.contract.account)
    submissions.pop(my_index)  
    trainers.pop(my_index)   

    honest_trainers, dishonest_trainers, _ = self.aggregator.assess_trainers(trainers, submissions)

    history = self.training_algo.fit(self.train_ds_batched)

    acc, loss = self.training_algo.test(self.test_ds_batched)
    # self._log_info(json.dumps({ 'event': 'self_agg_post', 'round': round,'ts': time.time_ns(), 'pre_acc': pre_acc, 'acc': acc, 'pre_loss': pre_loss, 'loss': loss }))

    trainingAccuracy = float_to_int(history.history["accuracy"][-1]*100)
    validationAccuracy = float_to_int(acc*100)

    top_trainers = ModelSelector(self.weights_loader).closestToLocal(self.training_algo.get_weights(), trainers, submissions)
    self._log_info(json.dumps({ 'event': 'self_agg_start', 'round': round, 'phase': phase.name, 'ts': time.time_ns(), 'top_trainers': top_trainers }))


    weights = self.training_algo.get_weights()
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
    self.last_action_completed = RoundPhase.WAITING_FOR_UPDATES

    self._log_info(json.dumps({ 'event': 'end', 'round': round, 'phase': phase.name, 'weights': weights_id, 'ts': time.time_ns(), 'submission': submission }))



  def train(self):
    # if not self.contract.is_selected_trainer():
    #   return

    (round, weights_id) = self.contract.get_training_round()
    phase = self.contract.get_round_phase()
    self._log_info(json.dumps({ 'event': 'start', 'round': round, 'phase': phase.name, 'ts': time.time_ns() }))
        
    if phase == RoundPhase.WAITING_FOR_FIRST_UPDATE:
      self.__do_first_update(phase)
    elif phase == RoundPhase.WAITING_FOR_PROOF_PRESENTMENT:
      self.__do_proof_presentment(phase)
    elif phase == RoundPhase.WAITING_FOR_UPDATES:
      self.__do_updates(round, phase, weights_id)
