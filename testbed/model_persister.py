from blocklearning.model_loaders.ipfs import IpfsModelLoader
from blocklearning.models import SimpleMLP
from blocklearning.training_algos import RegularAlgo
import click
import pickle
import tensorflow as tf

from blocklearning.weights_loaders.ipfs import IpfsWeightsLoader

@click.command()
@click.option('--location', default='./datasets', help='location for model')
@click.option('--image_lib', default='mnist', help='location for model')
@click.option('--idx', default='owner_val', help='location for model')
@click.option('--data_path', default='./datasets/mnist/10/{}.tfrecord', help='location of client data (tfrecs)')
def main(location, image_lib, idx, data_path):
    global_model = SimpleMLP.build(image_lib) 
    model_path = '{}/model_{}_{}.h5'.format(location, image_lib, idx)
    global_model.save(model_path)

    weights = global_model.get_weights()

    with open('{}/weights_{}_{}.pkl'.format(location, image_lib, idx), 'wb') as fp:
        pickle.dump(weights, fp)

    weights_loader = IpfsWeightsLoader(ipfs_api=None)
    cid_weights = weights_loader.store(weights)
    model_loader = IpfsModelLoader(None, None, None)
    cid_model = model_loader.store(model_path)
    print(cid_model, cid_weights)   

 
main()

