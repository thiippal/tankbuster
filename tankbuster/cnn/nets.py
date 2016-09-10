# Import the necessary packages
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential

# A class for various CNN architectures
class CNNArchitecture:
    def __init__(self):
        pass

    @staticmethod
    def select(architecture, *args, **kargs):

        # Map strings to functions
        nets = {
            "ShallowNet": CNNArchitecture.ShallowNet,
            "MiniVGGNet": CNNArchitecture.MiniVGGNet,
            "VGG16": CNNArchitecture.VGG16
        }

        # Initialize architecture
        net = nets.get(architecture, None)

        # Check if a net has been requested
        if net is None:
            return None

        # If the net is named correctly, return the network architecture
        return net(*args, **kargs)

    @staticmethod
    def ShallowNet(imgrows, imgcols, numchannels, numclasses, **kargs):
        # Initialize the model
        model = Sequential()

        # Add a CONV -> RELU layer
        model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(imgrows, imgcols, numchannels), dim_ordering="tf"))
        model.add(Activation("relu"))

        # Add a fully connected layer with a softmax classifier
        model.add(Flatten())
        model.add(Dense(numclasses))
        model.add(Activation("softmax"))

        # Return the network architecture
        return model

    @staticmethod
    def MiniVGGNet(imgrows, imgcols, numchannels, numclasses):
        # Initialize the model
        model = Sequential()

        # Define the first set of  CONV -> RELU -> CONV -> RELU -> POOL layers
        model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(imgrows, imgcols, numchannels), dim_ordering="tf"))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add dropout
        model.add(Dropout(0.25))

        # Define the second set of CONV -> RELU -> CONV -> RELU -> POOL layers
        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add dropout
        model.add(Dropout(0.25))

        # Define FC -> RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))

        # Add dropout
        model.add(Dropout(0.5))

        # Define the SoftMAX classifier
        model.add(Dense(numclasses))
        model.add(Activation("softmax"))

        # Return the network architecture
        return model

    @staticmethod
    def VGG16(imgrows, imgcols, numchannels, numclasses):
        # Initialize the model
        model = Sequential()

        # Add convolutional layers
        model.add(ZeroPadding2D((1, 1), input_shape=(imgrows, imgcols, numchannels)))
        model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering="tf"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Add fully connected layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(numclasses, activation='softmax'))

        return model
