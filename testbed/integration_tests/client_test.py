from blocklearning.trainer import Trainer
from blocklearning.weights_loaders import IpfsWeightsLoader
from blocklearning.models import SimpleMLP
import click

@click.command()
@click.option('--ipfs_api', default='None', help='api uri or None')
def main(ipfs_api):
    weight_loader = IpfsWeightsLoader(ipfs_api=ipfs_api)
    build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST
    smlp_global = SimpleMLP()
    model = smlp_global.build(build_shape, 10) 
    weights = model.get_weights()
    cid = weight_loader.store(weights)
    weights2 = weight_loader.load(cid)

    model.set_weights(weights2)
    # trainer = Trainer()

main()