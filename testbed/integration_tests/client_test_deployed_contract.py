import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from blocklearning.trainer import Trainer
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.models import SimpleMLP
import click
import tensorflow as tf
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from blocklearning.contract import Contract
import os
import logging
import utilities

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
    data_dir = '../' + data_dir
    accounts = utilities.read_json(os.path.join(data_dir, 'accounts.json'))
    account_address = list(accounts['owner'].keys())[0]
    account_password = accounts['owner'][account_address]
    return account_address, account_password

@click.command()
@click.option('--ipfs_api', default=None, help='api uri or None')
@click.option('--cid', default='QmfGcAp7mfxzrxSNNZmmtvpPAwToZ4Bbjz38v8ivKneLSb', help='api uri or None')
@click.option('--weights_path', default='../datasets/weights.pkl', help='location of weights .pkl file')
@click.option('--data_path', default='../datasets/mnist/5/train/1.tfrecord', help='location of client data (tfrecs)')
@click.option('--data_dir', default=utilities.default_datadir, help='ethereum data directory path')
@click.option('--provider', default='http://127.0.0.1:8545', help='web3 API HTTP provider')
@click.option('--abi', default='../../build/contracts/NoScore.json', help='contract abi file')
@click.option('--contract', required=True, help='contract address')


def main(ipfs_api, cid, weights_path, data_path, data_dir, provider, abi, contract):
    account_address, account_password = get_owner_account(data_dir)
    contract = Contract(log, provider, abi, account_address, account_password, contract)

    weights_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)

    if cid == '':
        cid = weights_loader.store(weights_path)
        print('weights cid', cid)

    weights = weights_loader.load(cid)

    build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST
    smlp_global = SimpleMLP()
    model = smlp_global.build(build_shape, 10) 

    model.set_weights(weights)
    train_ds = tf.data.experimental.load(data_path)
    trainer = Trainer(contract=contract, weights_loader=weights_loader, model=model, data=train_ds)
    trainer.train()

main()