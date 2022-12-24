import click
from unittest.mock import Mock
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.model_loaders import IpfsModelLoader
from blocklearning import Aggregator
from blocklearning.aggregators import FedAvgAggregator
import tensorflow as tf


@click.command()
@click.option('--image_lib', default='cifar', help='cifar or mnist')
@click.option('--ipfs_api', default=None, help='api uri or None')
@click.option('--model_cid', default='', help='model ipfs cid or empty string')
@click.option('--weights_cid', default='', help='weights ipfs cid or empty string')
@click.option('--model_path', default='../datasets/model_cifar_2.h5', help='location of model .h5 file')
@click.option('--weights_path', default='../datasets/weights_cifar.pkl', help='location of weights .pkl file')
@click.option('--train_data_path', default='../datasets/{}/10/train/{}.tfrecord', help='location of client data (tfrecs)')
@click.option('--test_data_path', default='../datasets/{}/10/test/{}.tfrecord', help='location of client data (tfrecs)')


def main(image_lib, ipfs_api, model_cid, weights_cid, model_path, weights_path, train_data_path, test_data_path):
    weights_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)
    contract = Mock()
    model_loader = IpfsModelLoader(contract=contract, weights_loader=weights_loader, ipfs_api=ipfs_api)

    if model_cid == '':
        model_cid = model_loader.store(model_path)
        print('model cid', model_cid)

    if weights_cid == '':
        weights_cid = weights_loader.store_from_path(weights_path)
        print('weights cid', weights_cid)
    
    contract.get_model.return_value = model_cid
    contract.get_weights.return_value = weights_cid
    model = model_loader.load()
    aggregator = FedAvgAggregator(weights_loader)
    train_ds = tf.data.experimental.load(train_data_path.format(image_lib, 9))
    test_ds = tf.data.experimental.load(test_data_path.format(image_lib, 9))

    aggregator = Aggregator(contract=contract, weights_loader=weights_loader, model=model, aggregator=aggregator, train_ds=train_ds, test_ds=test_ds)

    submission = (99933034181594841088, 99933034181594841088, 140, weights_cid)
    
    submissions = [submission]*2
    trainers = ['trainer']*2

    contract.get_submissions_for_aggregation.return_value = (9, trainers, submissions)
    aggregator.aggregate()
    new_weights_cid = contract.submit_aggregation.call_args.args
    print('new global weights cid', new_weights_cid)

main()