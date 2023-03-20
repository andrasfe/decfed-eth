import web3
import time
import click
import requests
import blocklearning
from blocklearning.model_loaders import IpfsModelLoader
import blocklearning.utilities as utilities
import tensorflow as tf
from blocklearning.contract import Contract
from blocklearning.pedersen import Pedersen
from blocklearning.trainers import PeerAggregatingTrainer
from blocklearning.aggregators import MultiKrumAggregator
from blocklearning.weights_loaders import IpfsWeightsLoader

@click.command()
@click.option('--provider', default='http://127.0.0.1:8545', help='web3 API HTTP provider')
@click.option('--ipfs', default='/ip4/127.0.0.1/tcp/5001', help='IPFS API provider')
@click.option('--abi', default='./build/contracts/Different.json', help='contract abi file')
@click.option('--pedersen_abi', default='../../build/contracts/ZKP/PedersenContract.json', help='pedersen contract abi file')
@click.option('--account', help='ethereum account to use for this computing server', required=True)
@click.option('--passphrase', help='passphrase to unlock account', required=True)
@click.option('--contract', help='contract address', required=True)
@click.option('--pedersen_contract', required=True, help='pedersen contract address')
@click.option('--log', help='logging file', required=True)
@click.option('--train', help='training data .tfrecord file', required=True)
@click.option('--test', help='training data .tfrecord file', required=True)
def main(provider, ipfs, abi, pedersen_abi, account, passphrase, contract, pedersen_contract, log, train, test):
  log = utilities.setup_logger(log, "client")

  if ipfs == 'None':
    ipfs = None

  weights_loader = IpfsWeightsLoader(ipfs)
  train_ds = tf.data.experimental.load(train)
  test_ds = tf.data.experimental.load(test)

  contract = Contract(log, provider, abi, account, passphrase, contract)
  pedersen_contract = Pedersen(log, provider, pedersen_abi, account, passphrase, pedersen_contract)

  # Load Model
  model_loader = IpfsModelLoader(contract, weights_loader, ipfs_api=ipfs)
  model = model_loader.load()

  trainer = PeerAggregatingTrainer(contract=contract, 
                                   pedersen=pedersen_contract,
                                   weights_loader=weights_loader, 
                                   model=model, 
                                   train_data=train_ds, 
                                   test_data=test_ds, 
                                   aggregator=MultiKrumAggregator(weights_loader, 10), 
                                   logger=log)

  while True:
    try:
      trainer.train()
    except web3.exceptions.ContractLogicError as err:
      print(err, flush=True)
    except requests.exceptions.ReadTimeout as err:
      print(err, flush=True)

    time.sleep(0.5)

main()
