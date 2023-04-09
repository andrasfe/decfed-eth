import web3
import time
import click
import requests
import blocklearning
from blocklearning.aggregators import MultiKrumAggregator
import blocklearning.model_loaders as model_loaders
from blocklearning.model_loaders.ipfs import IpfsModelLoader
from blocklearning.models.simple_model import SimpleMLP
import blocklearning.weights_loaders as weights_loaders
import blocklearning.utilities as utilities
from blocklearning.contract import RoundPhase
from blocklearning.aggregator import Aggregator

@click.command()
@click.option('--provider', default='http://127.0.0.1:8545', help='web3 API HTTP provider')
@click.option('--ipfs', default='/ip4/127.0.0.1/tcp/5001', help='IPFS API provider')
@click.option('--abi', default='./build/contracts/Different.json', help='contract abi file')
@click.option('--account', help='ethereum account to use for this computing server', required=True)
@click.option('--passphrase', help='passphrase to unlock account', required=True)
@click.option('--contract', help='contract address', required=True)
@click.option('--log', help='logging file', required=True)
def main(provider, ipfs, abi, account, passphrase, contract, log):

  if ipfs == 'None':
    ipfs = None

  log = utilities.setup_logger(log, "server")
  contract = blocklearning.Contract(log, provider, abi, account, passphrase, contract)
  weights_loader = weights_loaders.IpfsWeightsLoader(ipfs)

  model = SimpleMLP.build("mnist")
  server = Aggregator(contract=contract, 
                      weights_loader=weights_loader, 
                      model=model, 
                      train_ds=None, 
                      test_ds=None, 
                      aggregator=MultiKrumAggregator(weights_loader, 10), 
                      logger=log)

  while True:
    try:
      phase = contract.get_round_phase()
      if phase == RoundPhase.WAITING_FOR_AGGREGATIONS:
        server.aggregate()
    except web3.exceptions.ContractLogicError as err:
      print(err, flush=True)
    except requests.exceptions.ReadTimeout as err:
      print(err, flush=True)

    time.sleep(0.5)

main()
