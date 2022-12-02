from enum import Enum
from .base_contract import BaseContract
from web3 import Web3
import eth_abi


class Pedersen(BaseContract):
  def __init__(self, log, provider, abi_file, account, passphrase, contract_address):
    super().__init__(log, provider, abi_file, account, passphrase, contract_address)

  def get_commitment(self, random_t, value):
    return self.contract.functions.strCommit(random_t, value).call(self.default_opts)
