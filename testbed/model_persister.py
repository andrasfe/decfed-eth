from blocklearning.models import SimpleMLP
import click
import pickle
import tensorflow as tf

@click.command()
@click.option('--location', default='./datasets', help='location for model')
@click.option('--image_lib', default='cifar', help='location for model')
@click.option('--idx', default=5, help='location for model')
@click.option('--data_path', default='./datasets/cifar/10/train/{}.tfrecord', help='location of client data (tfrecs)')
def main(location, image_lib, idx, data_path):
    global_model = SimpleMLP.build(image_lib) 
    batched_ds = tf.data.experimental.load(data_path.format(idx))
    global_model.fit(batched_ds, epochs=3, verbose=True)
    global_model.save('{}/model_{}_{}.h5'.format(location, image_lib, idx))
    with open('{}/weights_{}_{}.pkl'.format(location, image_lib, idx), 'wb') as fp:
        weights = global_model.get_weights()
        pickle.dump(weights, fp)
        

 
main()

