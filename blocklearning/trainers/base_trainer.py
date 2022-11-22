from abc import ABC, abstractmethod
import json
import time

class BaseTrainer(ABC):  
    def __init__(self):
        self.__register()

    @abstractmethod
    def train(self):
        pass

    def _log_info(self, message):
        if self.logger is not None:
            self.logger.info(message)


    # Private utilities
    def __register(self):
        self._log_info(json.dumps({ 'event': 'checking_registration', 'ts': time.time_ns() }))

        self.contract.register_as_trainer()

        self._log_info(json.dumps({ 'event': 'registration_checked', 'ts': time.time_ns() }))
