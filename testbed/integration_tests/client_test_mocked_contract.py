from blocklearning.trainers import RegularTrainer
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.models import SimpleMLP
import click
import tensorflow as tf
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from blocklearning.contract import Contract
import pickle


@click.command()
@click.option('--ipfs_api', default=None, help='api uri or None')
@click.option('--image_lib', default='cifar', help='api uri or None')
@click.option('--cid', default='', help='api uri or None')
@click.option('--weights_path', default='../datasets/weights_{}.pkl', help='location of weights .pkl file')
@click.option('--data_path', default='../datasets/cifar/10/train/3.tfrecord', help='location of client data (tfrecs)')

def main(ipfs_api, image_lib, cid, weights_path, data_path):
    weights_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)

    if cid == '':
        cid = weights_loader.store(weights_path.format(image_lib))
        print('weights cid', cid)

    weights = weights_loader.load(cid)
    model = SimpleMLP.build(image_lib) 

    with open(weights, 'rb') as f:
        weight_values = pickle.load(f)
        model.set_weights(weight_values)
    contract = Mock()
    contract.get_training_round.return_value = (1, cid)
    train_ds = tf.data.experimental.load(data_path)
    trainer = RegularTrainer(contract=contract, weights_loader=weights_loader, model=model, data=train_ds)
    trainer.train()
    args = contract.submit_submission.call_args.args
    print(args)

main()