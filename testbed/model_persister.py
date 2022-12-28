from blocklearning.models import SimpleMLP
import click
import pickle
import tensorflow as tf

from blocklearning.weights_loaders.ipfs import IpfsWeightsLoader

@click.command()
@click.option('--location', default='./datasets', help='location for model')
@click.option('--image_lib', default='cifar', help='location for model')
@click.option('--idx', default='owner_val', help='location for model')
@click.option('--data_path', default='./datasets/cifar/10/{}.tfrecord', help='location of client data (tfrecs)')
def main(location, image_lib, idx, data_path):
    global_model = SimpleMLP.build(image_lib) 
    batched_ds = tf.data.experimental.load(data_path.format(idx))
    global_model.fit(batched_ds, epochs=3, verbose=True)
    global_model.save('{}/model_{}_{}.h5'.format(location, image_lib, idx))

    weights = global_model.get_weights()

    with open('{}/weights_{}_{}.pkl'.format(location, image_lib, idx), 'wb') as fp:
        pickle.dump(weights, fp)

    weights_loader = IpfsWeightsLoader(ipfs_api=None)
    cid = weights_loader.store(weights)
    print(cid)   

 
main()

