from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.model_loaders import IpfsModelLoader
import click

@click.command()
@click.option('--model_path', default='./datasets/model.h5', help='location of model .h5 file')
@click.option('--weights_path', default='./datasets/weights.pkl', help='location of weights .pkl file')

def main(model_path, weights_path):
    model_loader = IpfsModelLoader(contract=None, weights_loader=None, ipfs_api=None)
    model_cid = model_loader.store(model_path)
    weights_loader = IpfsWeightsLoader(ipfs_api=None)   
    weights_cid = weights_loader.store_from_path(weights_path)
    print('Update contract.json model with:', model_cid)
    print('Update contract.json weights with:', weights_cid)


main()