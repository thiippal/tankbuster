# Tankbuster

Tankbuster is a convolutional neural net (CNN) trained to classify images according to whether they contain tanks, or more specifically, <a href="https://en.wikipedia.org/wiki/T-72">Soviet/Russian T-72</a> main battle tanks, or not.

Built using <a href="http://keras.io">Keras</a>, the classifier has been trained using a collection of images showing T-72 from various angles against a collection of images showing street and natural scenes without tanks. Because appropriate images of T-72s are surprisingly scarce, the data has been augmented using the <a href="http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">ImageDataGenerator</a> class provided by Keras. The training data contains approximately 5000 images per class.

## Installation

Tankbuster may be installed from PyPI by typing the following command on the command line:

<code>pip install tankbuster</code>

## Usage

<code>python bust.py -i image.jpg</code>

<code>python bust.py -i directory/</code>

