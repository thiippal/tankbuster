# Import the necessary packages
from .. import cnn
from pkg_resources import resource_filename
from colorama import init, Fore
import numpy as np
import cv2

init(autoreset=True)  # Colorama switch for default output after printing coloured text

def bust(image):
    """
    Predict the class of the input image (other/t-72/bmp).

    Args:
        image: An image file.

    Returns:
        Prints the class label with the highest probability.
    """

    # Load and process the image
    original = cv2.imread(image)  # Load image
    resized = cv2.resize(original, (150, 150), interpolation=cv2.INTER_NEAREST)  # Resize image
    normalized = resized.astype("float") / 255.0  # Normalize the array into range 0...1
    reshaped = np.expand_dims(normalized, axis=0)  # Reshape for input for CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = cnn.CNNArchitecture.select('MiniVGGNet', 150, 150, 3, 3)  # Select MiniVGGNet
    model_weights = resource_filename(__name__, 'weights.h5')  # Locate model weights
    model.load_weights(model_weights)  # Load weights

    # Return the prediction
    predictions = model.predict(reshaped, verbose=0)[0]

    # Retrieve the most probable class prediction
    labels = {0: 'other', 1: 'T-72', 2: 'BMP'}  # Class labels
    pred = np.argmax(predictions)  # Most probable class label

    # Print the prediction and probabilities
    if pred == 1 or pred == 2:
        print (Fore.GREEN + "[POSITIVE] ") + (Fore.BLACK + "File {}: {} ({:.2f}%)").format(
            image, labels[int(pred)], predictions[pred] * 100)
    if pred == 0:
        print (Fore.RED + "[NEGATIVE] ") + (Fore.BLACK + "File {}: other ({:.2f}%)").format(
            image, predictions[pred] * 100)

def npbust(image, **kwargs):
    """
    Predict the class of the input image (other/t-72/bmp).

    Args:
        image: An RGB image as a NumPy array.

    Returns:
        Returns a dictionary of labels and their associated probabilities.
    """
    # Load and process the image
    original = cv2.resize(image, (150, 150), interpolation=cv2.INTER_NEAREST)  # Resize
    normalized = original.astype('float') / 255.0  # Normalize into range 0...1
    reshaped = np.expand_dims(normalized, axis=0)  # Reshape for input for CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = cnn.CNNArchitecture.select('MiniVGGNet', 150, 150, 3, 3)  # Select MiniVGGNet
    model_weights = resource_filename(__name__, 'weights.h5')  # Locate default weights
    if 'weights' in kwargs:  # Check if alternate weights have been provided
        model_weights = kwargs['weights']  # Set alternate weights
    model.load_weights(model_weights)  # Load weights

    # Return class probabilities
    predictions = model.predict(reshaped, verbose=0)[0]
    preds = {'other': (predictions[0] * 100), 't-72': (predictions[1] * 100), 'bmp': (predictions[2] * 100)}

    return preds
