from enum import Enum
from .base_contract import BaseContract


class RoundPhase(Enum):
  STOPPED = 0
  WAITING_FOR_UPDATES = 1
  WAITING_FOR_SCORES = 2
  WAITING_FOR_AGGREGATIONS = 3
  WAITING_FOR_TERMINATION = 4
  WAITING_FOR_BACKPROPAGATION = 5
  WAITING_FOR_FIRST_UPDATE = 6
  WAITING_FOR_PROOF_PRESENTMENT = 7

class Contract(BaseContract):
  def __init__(self, log, provider, abi_file, account, passphrase, contract_address):
    super().__init__(log, provider, abi_file, account, passphrase, contract_address)

  def get_model(self):
    return self.contract.functions.model().call(self.default_opts)

  def get_top_model(self):
    return self.get_model()

  def get_bottom_model(self):
    return self.contract.functions.bottomModel().call(self.default_opts)

  def get_weights(self, round):
    return self.contract.functions.weights(round).call(self.default_opts)

  def get_round(self):
    return self.contract.functions.round().call(self.default_opts)

  def get_round_phase(self):
    return RoundPhase(self.contract.functions.roundPhase().call(self.default_opts))

  def get_trainers(self):
    return self.contract.functions.getTrainers().call(self.default_opts)

  def get_aggregators(self):
    return self.contract.functions.getAggregators().call(self.default_opts)

  def get_scorers(self):
    return self.contract.functions.getScorers().call(self.default_opts)

  def get_training_round(self):
    [round, weights_cid] = self.contract.functions.getRoundForTraining().call(self.default_opts)
    return (round, weights_cid)

  def get_submissions_for_scoring(self):
    [round, trainers, submissions] = self.contract.functions.getUpdatesForScore().call(self.default_opts)
    return (round, trainers, submissions)

  def get_submissions_for_aggregation(self):
    [round, trainers, submissions] = self.contract.functions.getUpdatesForAggregation().call(self.default_opts)
    return (round, trainers, submissions)

  def get_submissions_for_round(self, selectedRound):
    [round, trainers, submissions] = self.contract.functions.getUpdatesForRound(selectedRound).call(self.default_opts)
    return (round, trainers, submissions)


  def get_scorings(self):
    [trainers, scorers, scores] = self.contract.functions.getScores().call(self.default_opts)
    return (trainers, scorers, scores)

  def get_gradient(self):
    return self.contract.functions.getGradient().call(self.default_opts)

  def register_as_trainer(self):
    self._unlock_account()
    if not self.contract.functions.registeredTrainers(self.account).call(self.default_opts):
      tx = self.contract.functions.registerTrainer().transact(self.default_opts)
      return tx, self._wait_tx(tx)

  def register_as_scorer(self):
    self._unlock_account()
    if not self.contract.functions.registeredScorers(self.account).call(self.default_opts):
      tx = self.contract.functions.registerScorer().transact(self.default_opts)
      return tx, self._wait_tx(tx)

  def register_as_aggregator(self):
    self._unlock_account()
    if not self.contract.functions.registeredAggregators(self.account).call(self.default_opts):
      tx = self.contract.functions.registerAggregator().transact(self.default_opts)
      return tx, self._wait_tx(tx)

  def submit_first_update(self, submission):
    self._unlock_account()
    tx = self.contract.functions.submitFirstUpdate(submission).transact(self.default_opts)
    return tx, self._wait_tx(tx)

  def validate_pedersen(self, random_t, hiddenWeights):
    self._unlock_account()
    tx = self.contract.functions.validatePedersen(random_t, hiddenWeights).transact(self.default_opts)
    return tx, self._wait_tx(tx)


  def submit_submission(self, submission):
    self._unlock_account()
    tx = self.contract.functions.submitUpdate(submission).transact(self.default_opts)
    return tx, self._wait_tx(tx)

  def submit_scorings(self, trainers, scores):
    self._unlock_account()
    tx = self.contract.functions.submitScores(trainers, scores).transact(self.default_opts)
    return tx, self._wait_tx(tx)

  def submit_aggregation(self, weights_id):
    self._unlock_account()
    tx = self.contract.functions.submitAggregation(weights_id).transact(self.default_opts)
    return tx, self._wait_tx(tx)

  def submit_aggregation_with_gradients(self, weights_id, trainers, gradients_ids):
    self._unlock_account()
    tx = self.contract.functions.submitAggregationWithGradients(weights_id, trainers, gradients_ids).transact(self.default_opts)
    return tx, self._wait_tx(tx)

  def confirm_backpropagation(self):
    self._unlock_account()
    tx = self.contract.functions.confirmBackpropagation().transact(self.default_opts)
    return tx, self._wait_tx(tx)

  def start_round(self, *args):
    self._unlock_account()
    tx = self.contract.functions.startRound(*args).transact(self.default_opts)
    return tx, self._wait_tx(tx)

  def terminate_round(self):
    self._unlock_account()
    tx = self.contract.functions.terminateRound().transact(self.default_opts)
    return tx, self._wait_tx(tx)

