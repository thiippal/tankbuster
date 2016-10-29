# Import the necessary packages
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential

# A class for various CNN architectures
class CNNArchitecture:
    def __init__(self):
        pass

    @staticmethod
    def select(architecture, *args, **kargs):

        # Map strings to functions
        nets = {
            "MiniVGGNet": CNNArchitecture.MiniVGGNet
        }

        # Initialize architecture
        net = nets.get(architecture, None)

        # Check if a net has been requested
        if net is None:
            return None

        # If the net is named correctly, return the network architecture
        return net(*args, **kargs)

    @staticmethod
    def MiniVGGNet(numchannels, imgrows, imgcols, numclasses):
        # Initialize the model
        model = Sequential()

        # Define the first set of  CONV -> RELU -> CONV -> RELU -> POOL layers
        model.add(Convolution2D(32, 3, 3, input_shape=(numchannels, imgrows, imgcols), dim_ordering='th'))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Add dropout
        model.add(Dropout(0.25))

        # Define the second set of CONV -> RELU -> CONV -> RELU -> POOL layers
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add dropout
        model.add(Dropout(0.35))

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