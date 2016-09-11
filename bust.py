import os
import argparse
import glob
from tankbuster.engine import bust

# Set up the argument parser
ap = argparse.ArgumentParser()

# Add argument for input
ap.add_argument("-i", "--input", required=True, help="The image or directory to be analysed.")

# Parse the arguments
args = vars(ap.parse_args())

# Assign arguments to variables
image = args['input']

# Check if the input is a directory
if os.path.isdir(image):

    # Feed the images to the CNN
    for images in glob.glob(str(image) + '*.*'):
        bust(images)

# Check if the input is a file
if os.path.isfile(image):

    # Feed the image to the CNN
    bust(image)