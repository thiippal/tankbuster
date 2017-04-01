import os
import argparse
from tankbuster.engine import bust
from keras.preprocessing.image import list_pictures

# Set up the argument parser
ap = argparse.ArgumentParser()

# Add argument for input
ap.add_argument("-i", "--input", required=True,
                help="The image or directory to be passed to the neural net.")
ap.add_argument("-n", "--network", required=True,
                help="The neural network to be used: 'ConvNet' or 'ResNet'.")

# Parse the arguments
args = vars(ap.parse_args())

# Assign arguments to variables
user_input = args['input']
network = args['network']

# Check if the input is a directory
if os.path.isdir(user_input):

    # Feed the images to the network
    images = list_pictures(user_input)

    # Loop over the images
    for i in images:

        # Feed images to the classifier to retrieve a dictionary of predictions
        preds = bust(i, network=network)

        # Get the prediction with the highest probability
        pred = max(preds, key=preds.get)

        # Print the prediction
        print "*** {} - predicted {} ({:.2f}%) ...".format(i,
                                                           pred,
                                                           preds[pred] * 100)

# Check if the input is a file
if os.path.isfile(user_input):

    # Feed images to the classifier to retrieve a dictionary of predictions
    preds = bust(user_input, network=network)

    # Get the prediction with the highest probability
    pred = max(preds, key=preds.get)

    # Print the prediction
    print "*** {} - predicted {} ({:.2f}%) ...".format(user_input,
                                                       pred,
                                                       preds[pred] * 100)
