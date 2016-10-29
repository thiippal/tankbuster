import os
import argparse
import glob
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
    images = list_images(user_input)  # list images
    for i in images:
        bust(i)

    # TODO Make this more robust by automatically excluding non-image files

# Check if the input is a file
if os.path.isfile(user_input):

    # Feed the image to the CNN
    bust(user_input)
