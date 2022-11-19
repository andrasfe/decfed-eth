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
@click.option('--ipfs_api', default=None, help='api uri or None')
@click.option('--cid', default='QmfGcAp7mfxzrxSNNZmmtvpPAwToZ4Bbjz38v8ivKneLSb', help='api uri or None')
@click.option('--weights_path', default='../datasets/weights.pkl', help='location of weights .pkl file')
@click.option('--data_path', default='../datasets/mnist/5/train/1.tfrecord', help='location of client data (tfrecs)')

def main(ipfs_api, cid, weights_path, data_path):
    weights_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)

    if cid == '':
        cid = weights_loader.store(weights_path)
        print('weights cid', cid)

    weights = weights_loader.load(cid)

    build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST
    smlp_global = SimpleMLP()
    model = smlp_global.build(build_shape, 10) 

    model.set_weights(weights)
    contract = Mock()
    contract.get_training_round.return_value = (1, cid)
    train_ds = tf.data.experimental.load(data_path)
    trainer = Trainer(contract=contract, weights_loader=weights_loader, model=model, data=train_ds)
    trainer.train()
    args = contract.submit_submission.call_args.args
    print(args)

main()