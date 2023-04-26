import numpy as np
import tensorflow as tf
import random

def load_mnist_data():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_images, train_labels

def get_random_samples(images, labels, num_samples=10):
    samples = []
    for _ in range(num_samples):
        index = random.randint(0, len(images) - 1)
        # Remove the flatten() function to keep the image as a 28x28 array
        image = images[index]
        samples.append(image)
    return samples

def main():
    images, labels = load_mnist_data()
    samples = get_random_samples(images, labels)
    print(samples[5])

if __name__ == '__main__':
    main()

