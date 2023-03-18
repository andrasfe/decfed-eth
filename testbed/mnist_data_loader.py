import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from imutils import paths
import random
import click
from sklearn.utils import shuffle

def load():
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28*28)
    test_data = test_data.reshape(test_data.shape[0], 28*28)

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.

    return train_data, test_data, train_labels, test_labels

def batch_data(data_shard, bs=30, flip=False):
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    data = np.array(data)
    
    labels = list(label)
    
    if flip:      
        labels = shuffle(labels)
    
    labels = np.array(labels)
    data = np.array(list(data))
    
    assert(data.shape[0] == labels.shape[0])
    
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    
    X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(data, 
                                                    label, 
                                                    train_size=0.9,
                                                    test_size=0.1, 
                                                    random_state=42)
        
    train_dataset = tf.data.Dataset.from_tensor_slices((np.array(list(X_data_train)), np.array(list(y_data_train))))
    test_dataset = tf.data.Dataset.from_tensor_slices((np.array(list(X_data_test)), np.array(list(y_data_test))))
    return (train_dataset.shuffle(len(y_data_train)).batch(bs, drop_remainder=True), test_dataset.batch(bs, drop_remainder=True))

def create_clients(image_list, label_list, num_clients=10, initial=''):
    #create a list of client names
    client_names = ['{}{}'.format(initial, i) for i in range(num_clients)]
    data = list(zip(image_list, label_list))
    size = len(data)//num_clients   
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    assert(len(shards) == len(client_names))
    return {client_names[i] : shards[i] for i in range(len(client_names))} 

@click.command()
@click.option('--no_trainers', default='10', help='number of clients')
@click.option('--data_path', default='./datasets', help='location of training data')
def main(no_trainers, data_path):
    no_trainers = int(no_trainers)

    image_list, _, label_list, _ = load()

    X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=42)   
    print(len(X_train), len(X_test), len(y_train), len(y_test)) 

    # create one extra for the owner data set
    participants = create_clients(X_train, y_train, num_clients=no_trainers + 1)

    path_template = "{}/mnist/{}/{}/{}.tfrecord"
    
    owner = list(participants.keys())[-1]

    for (client_name, data) in participants.items():
        train_ds, test_ds = batch_data(data)
        if client_name != owner:
            tf.data.experimental.save(train_ds, path_template.format(data_path, no_trainers, 'train', client_name))
            tf.data.experimental.save(test_ds, path_template.format(data_path, no_trainers, 'test', client_name))
        else:
            tf.data.experimental.save(test_ds, "{}/mnist/{}/owner_val.tfrecord".format(data_path, no_trainers))

main()
