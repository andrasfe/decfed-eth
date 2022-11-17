import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from imutils import paths
import click

def load(paths, verbose=-1):
    '''expects images for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels        
        im_gray = cv2.imread(imgpath , cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten() # cv2.imread(imgpath) 
        # print(image.shape)
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    return data, labels

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(data, 
                                                    label, 
                                                    train_size=0.8,
                                                    test_size=0.2, 
                                                    random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices((list(X_data_train), list(y_data_train)))
    test_dataset = tf.data.Dataset.from_tensor_slices((list(X_data_test), list(y_data_test)))
    return (train_dataset.shuffle(len(y_data_train)).batch(bs), test_dataset.batch(bs))

def create_clients(image_list, label_list, num_clients=5):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}'.format(i) for i in range(num_clients)]

    #randomize the data
    # data = list(zip(image_list, label_list))
    # random.shuffle(data)  # <- IID
    
    # sort data for non-iid
    max_y = np.argmax(label_list, axis=-1)
    sorted_zip = sorted(zip(max_y, label_list, image_list), key=lambda x: x[0])
    data = [(x,y) for _,y,x in sorted_zip]

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 

@click.command()
@click.option('--no_trainers', default='5', help='number of clients')
@click.option('--data_path', default='./datasets', help='location of training data')
@click.option('--img_path', default='./datasets/trainingSet', help='location of images')
def main(no_trainers, data_path, img_path):
    no_trainers = int(no_trainers)
    image_paths = list(paths.list_images(img_path))

    image_list, label_list = load(image_paths, verbose=10000)

    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)

    X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=42)   
    print(len(X_train), len(X_test), len(y_train), len(y_test)) 

    # create one extra for the owner data set
    participants = create_clients(X_train, y_train, num_clients=(no_trainers + 1))

    path_template = "{}/mnist/{}/{}/{}.tfrecord"
    
    owner = list(participants.keys())[-1]

    for (client_name, data) in participants.items():
        train_ds, test_ds = batch_data(data)
        if client_name != owner:
            tf.data.experimental.save(train_ds, path_template.format(data_path, no_trainers, 'train', client_name))
            tf.data.experimental.save(test_ds, path_template.format(data_path, no_trainers, 'test', client_name))
        else:
            tf.data.experimental.save(test_ds, "{}/mnist/{}/owner_val.tfrecord".format(data_path, no_trainers))

    # train_ds = tf.data.experimental.load(path_template.format(NO_TRAINERS, 'train', '1'))
    # print(train_ds)

   

main()
