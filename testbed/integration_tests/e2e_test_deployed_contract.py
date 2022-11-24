import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from blocklearning.trainers import RegularTrainer, PeerAggregatingTrainer
from blocklearning.aggregator import Aggregator
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.models import SimpleMLP
from blocklearning.aggregators import FedAvgAggregator, BasilAggregator
from blocklearning.contract import RoundPhase
import click
import tensorflow as tf
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

def get_account_by_role_by_index(data_dir, role, idx):
    data_dir = '../' + data_dir
    accounts = utilities.read_json(os.path.join(data_dir, 'accounts.json'))
    account_address = list(accounts[role].keys())[idx]
    account_password = accounts[role][account_address]
    return account_address, account_password

@click.command()
@click.option('--ipfs_api', default=None, help='api uri or None')
@click.option('--cid', default='QmfGcAp7mfxzrxSNNZmmtvpPAwToZ4Bbjz38v8ivKneLSb', help='api uri or None')
@click.option('--weights_path', default='../datasets/weights.pkl', help='location of weights .pkl file')
@click.option('--train_data_path', default='../datasets/mnist/5/train/{}.tfrecord', help='location of client data (tfrecs)')
@click.option('--test_data_path', default='../datasets/mnist/5/test/{}.tfrecord', help='location of client data (tfrecs)')
@click.option('--data_dir', default=utilities.default_datadir, help='ethereum data directory path')
@click.option('--provider', default='http://127.0.0.1:8545', help='web3 API HTTP provider')
@click.option('--abi', default='../../build/contracts/NoScore.json', help='contract abi file')
@click.option('--contract_address', required=True, help='contract address')


def main(ipfs_api, cid, weights_path, train_data_path, test_data_path, data_dir, provider, abi, contract_address):
    account_address, account_password = get_account_by_role_by_index(data_dir, 'owner', 0)
    contract = Contract(log, provider, abi, account_address, account_password, contract_address)

    weights_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)

    if cid == '':
        cid = weights_loader.store(weights_path)
        print('weights cid', cid)

    weights = weights_loader.load(cid)

    build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST
    smlp_global = SimpleMLP()
    model = smlp_global.build(build_shape, 10) 

    model.set_weights(weights)
    aggregator = FedAvgAggregator(-1, weights_loader)
    basil_aggregator = BasilAggregator(weights_loader)

    trainers = []
    local_contracts = []
    for i in range(5):
        account_address, account_password = get_account_by_role_by_index(data_dir, 'trainers', i)
        local_contract = Contract(log, provider, abi, account_address, account_password, contract_address)
        local_contracts.append(local_contract)
        local_model = tf.keras.models.clone_model(model)
        train_ds = tf.data.experimental.load(train_data_path.format(i))
        test_ds = tf.data.experimental.load(test_data_path.format(i))
        # trainer = RegularTrainer(contract=local_contract, weights_loader=weights_loader, model=local_model, data=train_ds)
        trainer = PeerAggregatingTrainer(contract=local_contract, weights_loader=weights_loader, model=local_model, 
                                        train_data=train_ds, test_data=test_ds, aggregator=basil_aggregator)
        trainers.append(trainer)


    account_address, account_password = get_account_by_role_by_index(data_dir, 'trainers', len(trainers))
    aggregator_contract = Contract(log, provider, abi, account_address, account_password, contract_address)
    server = Aggregator(aggregator_contract, weights_loader, model, aggregator, with_scores=False, logger=log)

    all_trainers = contract.get_trainers()
    all_aggregators = contract.get_aggregators()

    phase = contract.get_round_phase()
    round = contract.get_round()

    if phase == RoundPhase.STOPPED:
        contract.start_round(all_trainers, all_aggregators)
        round = contract.get_round()
        log.info('starting round {}'.format(round))


    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_UPDATES:
        for i in range(len(trainers)):
            try:
                trainers[i].train()
            except Exception as e:
                log.info('Trainer {}, Error: {} for round {}'.format(i, e, round))

    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_AGGREGATIONS:
        server.aggregate()
        log.info('terminated round {}'.format(round))


    phase = contract.get_round_phase()
    if phase == RoundPhase.WAITING_FOR_TERMINATION:
        contract.terminate_round()
        log.info('terminated round {}'.format(round))

    log.info('current state {}'.format(phase))
main()