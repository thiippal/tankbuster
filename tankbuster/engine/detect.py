# Import the necessary packages
import numpy as np
from PIL import Image
from .. import cnn
from pkg_resources import resource_filename
from colorama import init, Fore
init(autoreset=True)

def alpha_to_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # Needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def bust(image):
    # Load and process the image
    original = Image.open(image)
    if original.mode == 'RGBA':  # Check for alpha channel
        original = alpha_to_color(original)  # Remove alpha channel
    resized = original.resize((256, 256))  # Resize
    image_array = np.asarray(resized)  # Convert image to numpy array for rescaling
    rescaled = image_array.astype("float") / 255.0  # Scale into range 0...1
    reshaped = rescaled.reshape((1,) + rescaled.shape)  # Reshape for input to CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = cnn.CNNArchitecture.select('MiniVGGNet', 256, 256, 3, 2)  # Select MiniVGGNet
    model_weights = resource_filename(__name__, '../classifier/weights.hdf5')  # Locate model weights
    model.load_weights(model_weights)  # Load weights
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])  # Compile the model

    # Return the prediction
    prediction = model.predict_proba(reshaped, verbose=0)[0]

    # Check the predicted labels
    other = prediction[0]
    tank = prediction[1]

    if tank > 0.25:
        print (Fore.GREEN + "[POSITIVE] ") + (Fore.BLACK + "File {} contains a tank (confidence: {:.2f}%).").format(image, tank * 100)
    else:
        print (Fore.RED + "[NEGATIVE] ") + (Fore.BLACK + "File {} does not contain a tank (confidence: {:.2f}%).").format(image, other * 100)
