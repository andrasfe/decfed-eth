from blocklearning.models import SimpleMLP
import click
import pickle

@click.command()
@click.option('--location', default='./datasets', help='location for model')
@click.option('--image_lib', default='cifar', help='location for model')
def main(location, image_lib):
    global_model = SimpleMLP.build(image_lib) 
    global_model.save('{}/model_{}.h5'.format(location, image_lib))
    with open('{}/weights_{}.pkl'.format(location, image_lib), 'wb') as fp:
        weights = global_model.get_weights()
        pickle.dump(weights, fp)

 
main()

