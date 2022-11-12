from blocklearning.trainer import Trainer
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.models import SimpleMLP
import click
import tensorflow as tf
from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch
from blocklearning.contract import Contract


@click.command()
@click.option('--ipfs_api', default='None', help='api uri or None')
@click.option('--cid', default='', help='api uri or None')
@click.option('--data_path', default='../datasets/mnist/5/train/1.tfrecord', help='location of client data (tfrecs)')
# @patch('Contract.get_training_round')

def main(ipfs_api, cid, data_path):
    weight_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)
    build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST
    smlp_global = SimpleMLP()
    model = smlp_global.build(build_shape, 10) 

    if cid == '':
        weights = model.get_weights()
        cid = weight_loader.store(weights)
        print(cid)
    weights2 = weight_loader.load(cid)

    model.set_weights(weights2)
    contract = Mock()
    contract.get_training_round.return_value = (1, cid)
    train_ds = tf.data.experimental.load(data_path)
    trainer = Trainer(contract=contract, weights_loader=weight_loader, model=model, data=train_ds)
    trainer.train()

main()