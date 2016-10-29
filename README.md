# Tankbuster

Tankbuster is a convolutional neural net (CNN) trained to detect Soviet/Russian <a href="http://en.wikipedia.org/wiki/T-72">T-72</a> main battle tanks or <a href="http://en.wikipedia.org/wiki/BMP_development">BMP</a> armored personnel carriers in photographs.

Built using <a href="http://keras.io">Keras</a>, the classifier has been trained using a collection of images showing T-72s and BMPs from various angles against a collection of photographs featuring street and natural scenes. 

The data has been augmented using the Keras <a href="http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">ImageDataGenerator</a>, generating approximately 3600 images for each class.

The classifier achieves a 92.33% accuracy on a separate testing set.

## Installation

<b>Note: it is sincerely recommended to install Tankbuster into its own <a href="http://docs.python-guide.org/en/latest/dev/virtualenvs/">virtual environment</a>.</b> 

Assuming you have <a href="https://pip.pypa.io/en/stable/installing/">installed pip</a>, Tankbuster may be installed from the Python Package Index (PyPI) by typing the following command on the command line:

<code>pip install tankbuster</code>

This will also install the required modules: Pillow, Keras, colorama and h5py.

Naturally, you can also clone or download the <a href="https://github.com/thiippal/tankbuster">GitHub repository</a> to get the latest version. Enter the following command to install Tankbuster and the required packages:

<code>pip install .</code>

## Usage

### Running Tankbuster from the command line

If you wish to run Tankbuster from the command line, it is best to clone this repository to get the driver script, <code>bust.py</code>.

To examine a single image from the command line, enter the following command on the command line:

<code>python bust.py -i image.jpg</code>

To examine all images in a directory, use the command below. Take care to include the final slash.

<code>python bust.py -i directory/</code>

### Integrating Tankbuster into your own program

If you wish to integrate Tankbuster into your Python program, import the key functions of the module using:

<code>from tankbuster import bust, npbust</code>

This allows you to call the <i>bust</i> method, which takes an image file as input.

<code>bust('image.png')</code>

Alternatively, you can call the <i>npbust</i> method, which takes a NumPy array as input. This is particularly useful if your image processing pipeline relies on popular libraries such as OpenCV or Mahotas, which both make heavy use of NumPy.

<code>npbust(image)</code>

The <i>npbust</i> method returns a dictionary of class labels with their associated probabilities. These values can be used as the basis for further actions, such as flagging the image.

## In action

<b>Disembarking from a Ropucha-class landing craft</b><br>
Image contains a tank with 99.99% confidence.

<image src="demo_images/with_ropucha.jpg" width="400px">

<b>Parked next to a Lada</b><br>
Image contains a tank with 99.93% confidence.

<image src="demo_images/with_lada.jpg" width="400px">

<b>Featured in a low-quality screen capture</b><br>
Image contains a tank with 50.74% confidence.

<image src="demo_images/from_screen_capture.png" width="400px">
