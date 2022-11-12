from blocklearning.models import SimpleMLP
import click

@click.command()
@click.option('--location', default='./datasets', help='location for model')
def main(location):

    build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST

    smlp_global = SimpleMLP()
    global_model = smlp_global.build(build_shape, 10) 
    global_model.save('{}/model.h5'.format(location))
    global_model.save_weights('{}/weights.h5'.format(location))
 
main()

