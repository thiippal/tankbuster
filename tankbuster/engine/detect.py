# Import the necessary packages
import numpy as np
from .. import cnn
from pkg_resources import resource_filename
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

"""
This file, detect.py, contains the function that loads, pre-processes and feeds
the input image to the selected neural network. To make a prediction on a given
image, call the bust function.
"""


def bust(image, network):
    """
    Predict the class of the input image ('BMP', 'Other' or 'T-72').

    Arguments:
        image: A path to an image.

    Parameters:
        network: The network architecture used for making a prediction.

    Returns:
        A dictionary with labels and their associated softmax probabilities.
    """

    # Define a list of valid networks
    networks = ['ConvNet', 'ResNet']

    # Check if a valid network has been requested
    if network in networks:

        # If the request is valid, begin processing the image with the
        # following steps:

        # Load image
        if network == "ConvNet":
            original = load_img(image, target_size=(150, 150))
        if network == "ResNet":
            original = load_img(image, target_size=(224, 224))

        # Convert image into NumPy array and expand dimensions for input
        arr = img_to_array(original, data_format='channels_last')
        arr = np.expand_dims(arr, axis=0)

        # Normalize the image pixel values
        if network == "ConvNet":
            # For ConvNet, scale the pixel values into range [0..1]
            normalized = arr.astype("float") / 255.0
            # For ResNet, apply ImageNet preprocessing, i.e. subtract mean
        elif network == "ResNet":
            normalized = preprocess_input(arr, data_format='channels_last')

        # Select neural network architecture
        if network == "ConvNet":
            # Fetch architecture
            model = cnn.CNNArchitecture.select(network, 150, 150, 3, 3)
            # Locate ConvNet weights
            conv_weights = resource_filename(__name__, 'conv_weights.h5')
            # Load weights
            model.load_weights(conv_weights)

            # Return the prediction
            predictions = model.predict(normalized, verbose=0)[0]

            # Create a dictionary of predictions
            label_probs = {'bmp': (predictions[0]),
                           't-72': (predictions[1]),
                           'other': (predictions[2])}

            return label_probs

        elif network == "ResNet":
            # Fetch base model architecture
            base_model = cnn.CNNArchitecture.select(network, 224, 224, 3)

            # Extract features using ResNet50
            features = base_model.predict(normalized, batch_size=1, verbose=0)

            # Construct the top model
            top_model = cnn.CNNArchitecture.select("TopNet", features.shape[1:])

            # Locate weights for TopNet
            res_weights = resource_filename(__name__, 'res_weights.h5')
            # Load weights
            top_model.load_weights(res_weights)

            # Return the prediction
            predictions = top_model.predict(features, verbose=0)[0]

            # Create a dictionary of predictions
            label_probs = {'bmp': (predictions[0]),
                           't-72': (predictions[1]),
                           'other': (predictions[2])}

            return label_probs

    # If the requested network is not valid, quit and print an error message
    else:
        quit("'{}' is not a valid network. Terminating ...".format(network))
