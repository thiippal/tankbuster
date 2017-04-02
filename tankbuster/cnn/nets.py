# Import the necessary packages
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Sequential
from keras.regularizers import l2
from keras.applications import ResNet50
from keras.models import Model

"""
This file, nets.py, contains the architectures for the neural networks. These
architectures are created using the CNNArchitecture class and its select method.
"""


class CNNArchitecture:
    # A class for network architectures
    def __init__(self):
        pass

    @staticmethod
    def select(architecture, *args, **kargs):

        # Map strings to functions
        nets = {
            "ConvNet": CNNArchitecture.ConvNet,
            "ResNet": CNNArchitecture.ResNet,
            "TopNet": CNNArchitecture.TopNet
        }

        # Initialize architecture
        net = nets.get(architecture, None)

        # Check if a net has been requested
        if net is None:
            return None

        # If the net is named correctly, return the network architecture
        return net(*args, **kargs)

    @staticmethod
    def ConvNet(imgrows, imgcols, numchannels, numclasses):
        # Initialize the model
        model = Sequential()

        # Define the first convolutional block
        model.add(Conv2D(32, (3, 3), input_shape=(imgrows, imgcols, numchannels),
                         data_format='channels_last',
                         kernel_regularizer=l2(l=0.001)))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(l=0.001)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Define the second convolutional block
        model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l=0.001)))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l=0.001)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the feature maps
        model.add(Flatten())

        # Add dropout
        model.add(Dropout(rate=0.5))

        # Add fully-connected layer
        model.add(Dense(256, kernel_regularizer=l2(l=0.001)))
        model.add(Activation("relu"))

        # Add dropout
        model.add(Dropout(rate=0.5))

        # Define the SoftMAX classifier
        model.add(Dense(numclasses))
        model.add(Activation("softmax"))

        # Return the network architecture
        return model

    @staticmethod
    def ResNet(imgrows, imgcols, numchannels):
        # Initialize model without pre-trained weights
        resnet = ResNet50(include_top=False,
                          input_tensor=Input(shape=(imgrows,
                                                    imgcols,
                                                    numchannels)))

        # Get output from the average pooling layer
        model = Model(inputs=resnet.input,
                      outputs=resnet.get_layer('avg_pool').output)

        # Return the model
        return model

    @staticmethod
    def TopNet(input_tensor):
        model = Sequential()
        model.add(Flatten(input_shape=input_tensor))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        return model
