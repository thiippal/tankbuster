# Import the necessary packages
import numpy as np
from PIL import Image
from .. import cnn
from pkg_resources import resource_filename
from colorama import init, Fore
init(autoreset=True)

def alpha_to_color(image, colour=(255, 255, 255)):
    """
    Remove alpha channel from an image.

    Args:
        image: An image with an alpha channel.
        colour: Background colour of the new image.

    Returns:
        An image without alpha channel.
    """
    image.load()  # Needed for split()
    background = Image.new('RGB', image.size, colour)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def bust(image):
    """
    Predict the class of the input image (other/t-72/bmp).

    Args:
        image: An image file.

    Returns:
        Prints the class label with the highest probability.
    """
    # Load and process the image
    original = Image.open(image)
    if original.mode == 'RGBA':  # Check for alpha channel
        original = alpha_to_color(original)  # Remove alpha channel
    resized = original.resize((150, 150), resample=Image.BILINEAR)  # Resize
    image_array = np.asarray(resized)  # Convert image to numpy array for rescaling
    rescaled = image_array.astype("float") / 255.0  # Scale into range 0...1

    # Reorganize arrays for Theano
    reorganized = np.transpose(rescaled, (2, 0, 1))  # Moving array at index 2 (number of channels) to first position

    reshaped = reorganized.reshape((1,) + reorganized.shape)  # Reshape for input to CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = cnn.CNNArchitecture.select('MiniVGGNet', 3, 150, 150, 3)  # Select MiniVGGNet
    model_weights = resource_filename(__name__, 'weights.h5')  # Locate model weights
    model.load_weights(model_weights)  # Load weights
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])  # Compile the model

    # Return the prediction
    predictions = model.predict_proba(reshaped, verbose=0)[0]

    # Retrieve the most probable class prediction
    labels = {0: 'other', 1: 'T-72', 2: 'BMP'}  # Class labels
    pred = np.argmax(predictions)  # Most probable class label

    # Print the prediction and probabilities
    if pred == 1 or pred == 2:
        print (Fore.GREEN + "[POSITIVE] ") + (Fore.BLACK + "File {}: {} ({:.2f}%)").format(
            image, labels[int(pred)], predictions[pred] * 100)
    elif pred == 0:
        print (Fore.RED + "[NEGATIVE] ") + (Fore.BLACK + "File {}: other ({:.2f}%)").format(
            image, predictions[pred] * 100)

def npbust(image):
    """
    Predict the class of the input image (other/t-72/bmp).

    Args:
        image: An image as a NumPy array with three channels.

    Returns:
        Returns a dictionary of labels and their associated probabilities.
    """
    # Load and process the image
    original = Image.fromarray(image, 'RGB')
    resized = original.resize((150, 150), resample=Image.BILINEAR)  # Resize
    image_array = np.asarray(resized)  # Convert image to numpy array for rescaling
    rescaled = image_array.astype('float') / 255.0  # Scale into range 0...1
    reorganized = np.transpose(rescaled, (2, 0, 1))  # Moving array at index 2 (number of channels) to first position
    reshaped = reorganized.reshape((1,) + reorganized.shape)  # Reshape for input to CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = cnn.CNNArchitecture.select('MiniVGGNet', 3, 150, 150, 3)  # Select MiniVGGNet
    model_weights = resource_filename(__name__, '../classifier/weights.h5')  # Locate model weights
    model.load_weights(model_weights)  # Load weights
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])  # Compile the model

    # Return class probabilities
    predictions = model.predict_proba(reshaped, verbose=0)[0]
    preds = {'other': (predictions[0] * 100), 't-72': (predictions[1] * 100), 'bmp': (predictions[2] * 100)}

    return preds

