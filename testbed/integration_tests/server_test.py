import click
import tensorflow as tf
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from blocklearning.contract import Contract
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.model_loaders import IpfsModelLoader
from blocklearning import Aggregator
from blocklearning.aggregators import FedAvgAggregator


@click.command()
@click.option('--ipfs_api', default='None', help='api uri or None')
@click.option('--model_cid', default='', help='model ipfs cid or empty string')
@click.option('--weights_cid', default='', help='weights ipfs cid or empty string')
@click.option('--model_path', default='../datasets/model.h5', help='location of model .h5 file')
@click.option('--weights_path', default='../datasets/weights.pkl', help='location of weights .pkl file')
@click.option('--data_path', default='../datasets/mnist/5/train/1.tfrecord', help='location of client data (tfrecs)')
# @patch('Contract.get_training_round')

def main(ipfs_api, model_cid, weights_cid, model_path, weights_path, data_path):
    weights_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)
    contract = Mock()
    model_loader = IpfsModelLoader(contract=contract, weights_loader=weights_loader, ipfs_api=None)

    if model_cid == '':
        model_cid = model_loader.store(model_path)
        print('model cid', model_cid)

    if weights_cid == '':
        weights_cid = weights_loader.store(weights_path)
        print('weights cid', weights_cid)


    
    contract.get_model.return_value = model_cid
    contract.get_weights.return_value = weights_cid
    model = model_loader.load()
    aggregator = FedAvgAggregator(model.count, weights_loader)
    aggregator = Aggregator(contract, weights_loader, model, aggregator, with_scores=False, logger=None)

main()