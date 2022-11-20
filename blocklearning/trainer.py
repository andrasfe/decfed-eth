import json
import time
from .utilities import float_to_int
from tensorflow.keras.optimizers import SGD

class Trainer():
  def __init__(self, contract, weights_loader, model, data, logger = None, priv = None):
    self.logger = logger
    self.priv = priv
    self.weights_loader = weights_loader
    self.contract = contract
    self.train_ds_batched = data
    self.model = model
    self.__register()

  def train(self):
    (round, weights_id) = self.contract.get_training_round()

    if self.logger is not None:
      self.logger.info(json.dumps({ 'event': 'start', 'round': round, 'weights': weights_id, 'ts': time.time_ns() }))

    if weights_id != '':
      weights = self.weights_loader.load(weights_id)
      self.model.set_weights(weights)

      lr = 0.01 
      local_rounds = 5
      comms_round = 5
      loss='categorical_crossentropy'
      metrics = ['accuracy']
      optimizer = SGD(lr=lr, 
                      decay=lr / comms_round, 
                      momentum=0.9
                    )          

      self.model.compile(loss=loss, 
            optimizer=optimizer, 
            metrics=metrics)

    if self.logger is not None:
      self.logger.info(json.dumps({ 'event': 'train_start', 'round': round,'ts': time.time_ns() }))

    history = self.model.fit(self.train_ds_batched, epochs=local_rounds, verbose=True)

    if self.logger is not None:
      self.logger.info(json.dumps({ 'event': 'train_end', 'round': round,'ts': time.time_ns() }))

    trainingAccuracy = float_to_int(history.history["accuracy"][-1]*100)
    validationAccuracy = float_to_int(history.history["accuracy"][-1]*100)

    weights = self.model.get_weights()
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

    if self.logger is not None:
      self.logger.info(json.dumps({ 'event': 'end', 'round': round, 'weights': weights_id, 'ts': time.time_ns(), 'submission': submission }))

  # Private utilities
  def __register(self):
    if self.logger is not None:
      self.logger.info(json.dumps({ 'event': 'checking_registration', 'ts': time.time_ns() }))

    self.contract.register_as_trainer()

    if self.logger is not None:
      self.logger.info(json.dumps({ 'event': 'registration_checked', 'ts': time.time_ns() }))
