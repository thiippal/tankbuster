import os
import argparse
from tankbuster.engine import bust
from tankbuster.utils import list_images

# Set up the argument parser
ap = argparse.ArgumentParser()

# Add argument for input
ap.add_argument("-i", "--input", required=True, help="The image or directory to be analysed.")

# Parse the arguments
args = vars(ap.parse_args())

# Assign arguments to variables
user_input = args['input']

# Check if the input is a directory
if os.path.isdir(user_input):

    # Feed the images to the CNN
    images = list_images(user_input)  # retrieve image files
    for i in images:  # loop over the images
        bust(i)  # feed image to the classifier

# Check if the input is a file
if os.path.isfile(user_input):

    # Feed the image to the classifier
    bust(user_input)
