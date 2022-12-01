import json
import time
from enum import Enum
from web3 import Web3
from web3.middleware import geth_poa_middleware

def get_web3(provider, abi_file, account, passphrase, contract):
  provider = Web3.HTTPProvider(provider)
  abi = get_abi(abi_file)

  web3 = Web3(provider)
  web3.middleware_onion.inject(geth_poa_middleware, layer=0)
  web3.geth.personal.unlock_account(account, passphrase, 600)

  contract = web3.eth.contract(address=contract, abi=abi)
  defaultOpts = { 'from': account }

  return (web3, contract, defaultOpts)

def get_abi(filename):
  with open(filename) as file:
    contract_json = json.load(file)
  return contract_json['abi']

class BaseContract():
  def __init__(self, log, provider, abi_file, account, passphrase, contract_address):
    self.log = log
    self.account = account
    self.passphrase = passphrase
    (web3, contract, default_opts) = get_web3(provider, abi_file, account, passphrase, contract_address)
    self.web3 = web3
    self.contract = contract
    self.default_opts = default_opts


  def _wait_tx(self, tx):
    self.log.info(json.dumps({ 'event': 'tx_start', 'tx': tx.hex(), 'ts': time.time_ns() }))
    receipt = self.web3.eth.wait_for_transaction_receipt(tx)
    self.log.info(json.dumps({ 'event': 'tx_end', 'tx': tx.hex(), 'gas': receipt.gasUsed, 'ts': time.time_ns() }))
    return receipt

  def _unlock_account(self):
    self.web3.geth.personal.unlock_account(self.account, self.passphrase, 600)
