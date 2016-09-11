# Tankbuster

Tankbuster is a convolutional neural net (CNN) trained to classify images according to whether they contain <a href="https://en.wikipedia.org/wiki/T-72">Soviet/Russian T-72</a> main battle tanks or not.

Built using <a href="http://keras.io">Keras</a>, the classifier has been trained using a collection of images showing T-72 from various angles against a collection of images showing both street and natural scenes without tanks. The data has been augmented using the Keras <a href="http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">ImageDataGenerator</a> class, resulting in approximately 5000 images per class.

## Installation

Tankbuster may be installed from the Python Package Index (PyPI) by typing the following command on the command line:

<code>pip install tankbuster</code>

## Usage

### Running Tankbuster from the command line

To examine a single image from the command line, enter the following command on the command line:

<code>python bust.py -i image.jpg</code>

To examine all images in a directory, use the command below. Take care to include the final slash.

<code>python bust.py -i directory/</code>

### Integrating Tankbuster into your code

If you wish to integrate Tankbuster into your code, import the module using:

<code>import tankbuster</code>

Importing the module allows you to call the <i>bust</i> method, which takes an image file as the input.

<code>tankbuster.bust('image.png')</code>
