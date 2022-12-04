import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from blocklearning.trainers import RegularTrainer, PeerAggregatingTrainer
from blocklearning.aggregator import Aggregator
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.models import SimpleMLP
from blocklearning.aggregators import FedAvgAggregator, BasilAggregator
from blocklearning.contract import RoundPhase
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

def get_account_by_role_by_index(data_dir, role, idx):
    data_dir = '../' + data_dir
    accounts = utilities.read_json(os.path.join(data_dir, 'accounts.json'))
    account_address = list(accounts[role].keys())[idx]
    account_password = accounts[role][account_address]
    return account_address, account_password


@click.command()
@click.option('--data_dir', default=utilities.default_datadir, help='ethereum data directory path')
@click.option('--provider', default='http://127.0.0.1:8545', help='web3 API HTTP provider')
@click.option('--pedersen_abi', default='../../build/contracts/PedersenContract.json', help='pedersen contract abi file')
@click.option('--pedersen_address', required=True, help='pedersen contract address')


def main(data_dir, provider, pedersen_abi, pedersen_address):
        account_address, account_password = get_account_by_role_by_index(data_dir, 'owner', 0)
        pedersen = Pedersen(log=log, provider=provider, abi_file=pedersen_abi, account=account_address, passphrase=account_password, contract_address=pedersen_address)
        random_T = 62625621
        hiddenWeights = 'Qmaw8ds9SDr6L83o5EJHtMVDkvzvz2Rou7zpa4CRAUnWLs'
        # commitment = [1393670059408418003925313476398630467342153378579412997904062059614727615557, 
        # 95405997805013254323017304745703196371164049587652558912277866144378897820336]
        commitment = pedersen.get_commitment(random_t=random_T, value=hiddenWeights)
        valid = pedersen.verify(random_T, hiddenWeights, commitment)
        assert(valid)

main()