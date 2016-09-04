# Tankbuster

Tankbuster is a convolutional neural net (CNN) trained to classify images according to whether they contain a tank, or more specifically, a <a href="https://en.wikipedia.org/wiki/T-72">Soviet/Russian T-72</a> main battle tank, or not.

Built using <a href="http://keras.io">Keras</a>, the classifier has been trained using a collection of images showing T-72 from various angles against a collection of images showing street and natural scenes without tanks. Because appropriate images of T-72s are surprisingly scarce, the data has been augmented using the <a href="http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html">ImageDataGenerator</a> class provided by Keras. Both data sets used to train the classifier contain approximately 5000 images.
