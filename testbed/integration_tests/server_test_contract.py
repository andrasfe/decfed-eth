import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from blocklearning.trainers import RegularTrainer, PeerAggregatingTrainer
from blocklearning.aggregator import Aggregator
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.models import SimpleMLP
from blocklearning.aggregators import FedAvgAggregator, BasilAggregator
from blocklearning.contract import RoundPhase
from blocklearning.training_algos import RegularAlgo
from blocklearning.diffpriv import Gaussian
from web3.exceptions import  ContractLogicError
import click
import tensorflow as tf
from blocklearning.contract import Contract
from blocklearning.pedersen import Pedersen
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

def get_account_by_role_by_index(data_dir, role, idx):
    data_dir = '../' + data_dir
    accounts = utilities.read_json(os.path.join(data_dir, 'accounts.json'))
    account_address = list(accounts[role].keys())[idx]
    account_password = accounts[role][account_address]
    return account_address, account_password

def run_all_trainers(trainers):
    for i in range(len(trainers)):
        try:
            trainers[i].train()
        except Exception as e:
            log.info('Trainer {}, Error: {} for round {}'.format(i, e, round))


@click.command()
@click.option('--ipfs_api', default=None, help='api uri or None')
@click.option('--cid', default='', help='api uri or None')
@click.option('--image_lib', default='cifar', help='cifar or mnist')
@click.option('--weights_path', default='../datasets/weights.pkl', help='location of weights .pkl file')
@click.option('--train_data_path', default='../datasets/{}/10/train/{}.tfrecord', help='location of client data (tfrecs)')
@click.option('--test_data_path', default='../datasets/{}/10/test/{}.tfrecord', help='location of client data (tfrecs)')
@click.option('--owner_data_path', default='../datasets/{}/10/owner_val.tfrecord', help='location of client data (tfrecs)')
@click.option('--data_dir', default=utilities.default_datadir, help='ethereum data directory path')
@click.option('--provider', default='http://127.0.0.1:8545', help='web3 API HTTP provider')
@click.option('--abi', default='../../build/contracts/Different.json', help='contract abi file')
@click.option('--pedersen_abi', default='../../build/contracts/PedersenContract.json', help='pedersen contract abi file')
@click.option('--contract_address', required=True, help='contract address')
@click.option('--pedersen_address', required=True, help='pedersen contract address')


def main(ipfs_api, cid, image_lib, weights_path, train_data_path, test_data_path, owner_data_path, data_dir, provider, abi, pedersen_abi, contract_address, pedersen_address):
    account_address, account_password = get_account_by_role_by_index(data_dir, 'owner', 0)
    contract = Contract(log, provider, abi, account_address, account_password, contract_address)

    weights_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)
    model = SimpleMLP.build(image_lib) 

    # model.set_weights(weights)
    aggregator = FedAvgAggregator(weights_loader)

    account_address, account_password = get_account_by_role_by_index(data_dir, 'trainers', 10)
    aggregator_contract = Contract(log, provider, abi, account_address, account_password, contract_address)
    train_ds = tf.data.experimental.load(train_data_path.format(image_lib, 9))
    test_ds = tf.data.experimental.load(test_data_path.format(image_lib, 9))
    server = Aggregator(contract=aggregator_contract, weights_loader=weights_loader, model=model, train_ds=train_ds, test_ds=test_ds, aggregator=aggregator, logger=log)

    try:
        server.aggregate()
    except ContractLogicError:
        print('this is expected')

 
    # round = contract.get_round()
    # (_, trainers, submissions) = aggregator_contract.get_submissions_for_round(round - 1)

    # for submission in submissions:
    #     cid = submission[3]
    #     weights = weights_loader.load(cid)
    #     model.set_weights(weights) 
    #     algo = RegularAlgo(model, 2, True)
    #     acc, loss = algo.test(test_ds)
    #     benchmarks='{},{},{}'.format(round, acc, loss)
    #     print(benchmarks)


    # cid = contract.get_weights_for_round(round - 1)
    # print(cid)

    # global_weights = weights_loader.load(cid)
    # model.set_weights(global_weights)


    # log.info(cid)
    
main()