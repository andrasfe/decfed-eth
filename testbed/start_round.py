# Hack path so we can load 'blocklearning'
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import logging
import time
import random
import json
import utilities as utilities
import blocklearning
import blocklearning.model_loaders as model_loaders
import blocklearning.weights_loaders as weights_loaders
import blocklearning.utilities as butilities
import tensorflow as tf
from blocklearning.contract import RoundPhase

# Setup Log
log_file = '../../blocklearning-results/results/CURRENT/logs/manager.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
if os.path.exists(log_file):
  print('a log already exists, please make sure to save the previous logs')
  # exit(1)

logging.basicConfig(level="INFO", handlers=[
  logging.StreamHandler(),
  logging.FileHandler(filename=log_file, encoding='utf-8', mode='a+')
])
log = logging.getLogger("manager")

def get_owner_account(data_dir):
  accounts = utilities.read_json(os.path.join(data_dir, 'accounts.json'))
  account_address = list(accounts['owner'].keys())[0]
  account_password = accounts['owner'][account_address]
  return account_address, account_password

@click.command()
@click.option('--provider', default='http://127.0.0.1:8545', help='web3 API HTTP provider')
@click.option('--abi', default='../build/contracts/Different.json', help='contract abi file')
@click.option('--contract', required=True, help='contract address')
@click.option('--data-dir', default=utilities.default_datadir, help='ethereum data directory path')
@click.option('--rounds', default=1, type=click.INT, help='number of rounds')
@click.option('--ipfs_api', default='none', help='whether to use API or not')
def main(provider, abi, contract, data_dir,rounds, ipfs_api):
  account_address, account_password = get_owner_account(data_dir)
  contract = blocklearning.Contract(log, provider, abi, account_address, account_password, contract)

  if ipfs_api == 'none':
    ipfs_api = None

  weights_loader = weights_loaders.IpfsWeightsLoader(ipfs_api=ipfs_api)
  model_loader = model_loaders.IpfsModelLoader(contract, weights_loader, ipfs_api=ipfs_api)
  model = model_loader.load()
 
  all_trainers = contract.get_trainers()
  all_aggregators = contract.get_aggregators()

  phase = contract.get_round_phase()
  round = contract.get_round()

  if phase == RoundPhase.STOPPED:
      contract.start_round(all_trainers, all_aggregators, 15000)
      round = contract.get_round()
      log.info('starting round {}'.format(round))

  log.info(json.dumps({ 'event': 'end', 'ts': time.time_ns(), 'round': round, 'weights': weights, 'accuracy': butilities.float_to_int(accuracy) }))

main()
