from blocklearning.models import SimpleMLP
import click
import pickle

@click.command()
@click.option('--location', default='./datasets', help='location for model')
def main(location):

    build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST

    smlp_global = SimpleMLP()
    global_model = smlp_global.build(build_shape, 10) 
    global_model.save('{}/model.h5'.format(location))
    with open('{}/weights.pkl'.format(location), 'wb') as fp:
        weights = global_model.get_weights()
        pickle.dump(weights, fp)

 
main()

