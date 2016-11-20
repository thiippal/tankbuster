# Import the necessary packages
import numpy as np
from PIL import Image
from .. import cnn
from pkg_resources import resource_filename
from colorama import init, Fore
from keras.preprocessing.image import load_img, img_to_array
init(autoreset=True)

def bust(image):
    """
    Predict the class of the input image (other/t-72/bmp).

    Args:
        image: An image file.

    Returns:
        Prints the class label with the highest probability.
    """

    # Load and process the image
    original = load_img(image, target_size=(150, 150))  # Load image and resize to 150x150
    array = img_to_array(original, dim_ordering='tf')  # Convert image to numpy array
    normalized = array.astype("float") / 255.0  # Normalize the array into range 0...1
    reshaped = normalized.reshape((1,) + normalized.shape)  # Reshape for input for CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = cnn.CNNArchitecture.select('MiniVGGNet', 150, 150, 3, 3)  # Select MiniVGGNet
    model_weights = resource_filename(__name__, 'tf-weights.h5')  # Locate model weights
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
        image: An RGB image as a NumPy array.

    Returns:
        Returns a dictionary of labels and their associated probabilities.
    """
    # Load and process the image
    original = Image.fromarray(image, 'RGB')
    resized = original.resize((150, 150), resample=Image.BILINEAR)  # Resize
    image_array = np.asarray(resized)  # Convert image to numpy array for rescaling
    normalized = image_array.astype('float') / 255.0  # Normalize into range 0...1
    reshaped = normalized.reshape((1,) + normalized.shape)  # Reshape for input to CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = cnn.CNNArchitecture.select('MiniVGGNet', 150, 150, 3, 3)  # Select MiniVGGNet
    model_weights = resource_filename(__name__, 'tf-weights.h5')  # Locate model weights
    model.load_weights(model_weights)  # Load weights
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])  # Compile the model

    # Return class probabilities
    predictions = model.predict_proba(reshaped, verbose=0)[0]
    preds = {'other': (predictions[0] * 100), 't-72': (predictions[1] * 100), 'bmp': (predictions[2] * 100)}

    return preds
