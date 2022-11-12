
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
    
    @staticmethod
    def build_adv(shape, classes):
        model = Sequential()
        model.add(Input(shape=(shape[0], shape[1], shape[2])))
        #model.add(Lambda(lambda x: expand_dims(x, axis=-1)))
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Activation("relu"))
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Activation("relu"))
        model.add(Conv2D(filters=512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model