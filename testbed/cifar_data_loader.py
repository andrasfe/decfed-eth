import numpy as np
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import random
import click

from tensorflow.keras.datasets import cifar10

def batch_data(data_shard, bs=5):
    data, label = zip(*data_shard)
    X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(data, 
                                                    label, 
                                                    train_size=0.8,
                                                    test_size=0.2, 
                                                    random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices((list(X_data_train), list(y_data_train)))
    test_dataset = tf.data.Dataset.from_tensor_slices((list(X_data_test), list(y_data_test)))
    return (train_dataset.shuffle(len(y_data_train)).batch(bs), test_dataset.batch(bs))


def create_clients(image_list, label_list, num_clients=10):
    #create a list of client names
    client_names = ['{}'.format(i) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)  # <- IID
    
    # sort data for non-iid
    max_y = np.argmax(label_list, axis=-1)
    sorted_zip = sorted(zip(max_y, label_list, image_list), key=lambda x: x[0])
    data = [(x,y) for _,y,x in sorted_zip]

    shards = []
    vals = None
    k_nums = [0]

    while min(k_nums) <3000:
        vals = np.random.default_rng().dirichlet(np.ones(num_clients), size=1)
        k_nums = [round(v) for v in vals[0]*len(data)]

    offset = 0

    for i in range(num_clients):
        end = len(data) if i == num_clients - 1 else offset + k_nums[i]
        shards.append(data[offset : end])
        offset = end + 1

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))
    assert(sum([len(shard) for shard in shards]) <= len(data))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 

@click.command()
@click.option('--no_trainers', default=10, help='number of clients')
@click.option('--data_path', default='./datasets', help='location of training data')

def main(no_trainers, data_path):
    try:
        shutil.rmtree('{}/cifar'.format(data_path))
    except:
        print('directory {}/cifar does not exist'.format(data_path))

    (X_train,y_train) , (X_test, y_test) = cifar10.load_data()

    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    X_train = X_train/255
    X_test = X_test/255

    y_train  = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # create one extra for the owner data set
    participants = create_clients(X_train, y_train, num_clients=(no_trainers + 1))

    path_template = "{}/cifar/{}/{}/{}.tfrecord"
    
    owner = list(participants.keys())[-1]

    for (client_name, data) in participants.items():
        train_ds, test_ds = batch_data(data)
        if client_name != owner:
            tf.data.experimental.save(train_ds, path_template.format(data_path, no_trainers, 'train', client_name))
            tf.data.experimental.save(test_ds, path_template.format(data_path, no_trainers, 'test', client_name))
        else:
            tf.data.experimental.save(test_ds, "{}/cifar/{}/owner_val.tfrecord".format(data_path, no_trainers))

main()