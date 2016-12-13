# Import the necessary packages
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Layer, ZeroPadding2D
from keras.models import Sequential
from keras import backend as K

# A class for various CNN architectures
class CNNArchitecture:
    def __init__(self):
        pass

    @staticmethod
    def select(architecture, *args, **kargs):

        # Map strings to functions
        nets = {
            "MiniVGGNet": CNNArchitecture.MiniVGGNet,
            "MiniVGGNetFC": CNNArchitecture.MiniVGGNetFC
        }

        # Initialize architecture
        net = nets.get(architecture, None)

        # Check if a net has been requested
        if net is None:
            return None

        # If the net is named correctly, return the network architecture
        return net(*args, **kargs)

    @staticmethod
    def MiniVGGNet(imgrows, imgcols, numchannels, numclasses):
        # Initialize the model
        model = Sequential()

        # Define the first set of  CONV -> RELU -> CONV -> RELU -> POOL layers
        model.add(Convolution2D(32, 3, 3, input_shape=(imgrows, imgcols, numchannels), dim_ordering='tf'))
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

    @staticmethod
    def MiniVGGNetFC(imgrows, imgcols, numchannels, numclasses):
        # Initialize the model
        model = Sequential()

        # Define the first set of  CONV -> RELU -> CONV -> RELU -> POOL layers
        model.add(Convolution2D(32, 3, 3, input_shape=(imgrows, imgcols, numchannels), dim_ordering='tf'))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # Define the second set of CONV -> RELU -> CONV -> RELU -> POOL layers
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Define the fully convolutional block
        model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(Convolution2D(512, 5, 5, activation='relu', name='dense_1'))
        model.add(Convolution2D(3, 1, 1, activation='relu', name='dense_2'))

        # Define the 4D SoftMAX classifier
        model.add(Softmax4D(axis=1))

        # Return the network architecture
        return model


class Softmax4D(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape