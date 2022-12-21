
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPool2D

class SimpleMLP:

    @staticmethod
    def build_mnist():
        model = Sequential()
        model.add(Dense(200, input_shape=(784,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        return model
    
    @staticmethod
    def build_cifar():
        model = Sequential()

        model.add(Conv2D(filters = 32, kernel_size = (4,4), input_shape = (32, 32, 3), activation = "relu"))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Conv2D(filters = 64, kernel_size = (4,4), input_shape = (32, 32, 3), activation = "relu"))
        model.add(MaxPool2D(pool_size = (2,2)))

        model.add(Flatten())

        model.add(Dense(512, activation = "relu"))
        model.add(Dense(256, activation = "relu"))
        model.add(Dense(128, activation = "relu"))

        model.add(Dense(10, activation = "softmax"))

        model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
        return model

    @staticmethod
    def build(image_lib):
        if image_lib == 'mnist':
            return SimpleMLP.build_mnist()
        else:
            return SimpleMLP.build_cifar()

