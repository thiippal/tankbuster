# Tankbuster

Tankbuster is a convolutional neural net trained to classify images according to whether they contain a tank, or more specifically, a Soviet/Russian T-72 main battle tank.

Built using Keras, the classifier has been trained using a collection of images showing T-72 from various angles, against a collection of images showing street and natural scenes without tanks. Because good images of T-72s are somewhat scarce, the data has been augmented using the ImageDataGenerator class provided with Keras. Both data sets contain approximately 5000 images.
