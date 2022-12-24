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

    if cid == '':
        with open(weights_path, 'rb') as fp:
            weights = pickle.load(fp)
            cid = weights_loader.store(weights)
            print('weights cid', cid)

    weights = weights_loader.load(cid)
    model = SimpleMLP.build(image_lib) 

    # model.set_weights(weights)
    aggregator = FedAvgAggregator(weights_loader)
    basil_aggregator = BasilAggregator(weights_loader)
    priv = Gaussian()

    trainers = []
    local_contracts = []
    for i in range(10):
        account_address, account_password = get_account_by_role_by_index(data_dir, 'trainers', i)
        local_contract = Contract(log, provider, abi, account_address, account_password, contract_address)
        pedersen_contract = Pedersen(log, provider, pedersen_abi, account_address, account_password, pedersen_address)
        local_contracts.append(local_contract)
        local_model = tf.keras.models.clone_model(model)
        train_ds = tf.data.experimental.load(train_data_path.format(image_lib,i))
        test_ds = tf.data.experimental.load(test_data_path.format(image_lib, i))
        # trainer = RegularTrainer(contract=local_contract, weights_loader=weights_loader, model=local_model, data=train_ds)
        trainer = PeerAggregatingTrainer(contract=local_contract, pedersen=pedersen_contract, weights_loader=weights_loader, model=local_model, 
                                        train_data=train_ds, test_data=test_ds, aggregator=basil_aggregator, priv=None)
        trainers.append(trainer)

    aggregator_idx = len(trainers)
    account_address, account_password = get_account_by_role_by_index(data_dir, 'trainers', aggregator_idx)
    aggregator_contract = Contract(log, provider, abi, account_address, account_password, contract_address)
    aggregator_model = tf.keras.models.clone_model(model)
    train_ds = None # for now
    test_ds = tf.data.experimental.load(owner_data_path.format(image_lib))

    server = Aggregator(contract=aggregator_contract, weights_loader=weights_loader, model=aggregator_model, 
                                        train_ds=train_ds, test_ds=test_ds, aggregator=aggregator, with_scores=False, logger=log)

    all_trainers = contract.get_trainers()
    all_aggregators = contract.get_aggregators()

    phase = contract.get_round_phase()
    round = contract.get_round()

    if phase == RoundPhase.STOPPED:
        contract.start_round(all_trainers, all_aggregators, 15000)
        round = contract.get_round()
        log.info('starting round {}'.format(round))


    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_FIRST_UPDATE:
        run_all_trainers(trainers)

    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_PROOF_PRESENTMENT:
        run_all_trainers(trainers)

    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_UPDATES:
        run_all_trainers(trainers)


    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_AGGREGATIONS:
        server.aggregate()
        log.info('terminated round {}'.format(round))


    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_TERMINATION:
        contract.terminate_round()
        log.info('terminated round {}'.format(round))

    log.info('current state {}'.format(phase))

    (_, trainers, submissions) = contract.get_submissions_for_round(round)

    for submission in submissions:
        subm_benchmark = '{},{},{}'.format(round, submission[0], submission[1])
        print(subm_benchmark)
        log.info(subm_benchmark)

    # cid = contract.get_weights_for_round(round)
    # global_weights = weights_loader.load(cid)
    # model.set_weights(global_weights)
    # test_ds = tf.data.experimental.load(owner_data_path)
    # algo = RegularAlgo(model, 2, True)
    # acc, loss = algo.test(test_ds)
    # benchmarks='{},{},{}'.format(round, acc, loss)
    # print(benchmarks)
    # log.info(benchmarks)
    
main()