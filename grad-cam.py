from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tankbuster.cnn import CNNArchitecture
from tankbuster.engine import npbust
import keras.backend as K
import tensorflow as tf
import numpy as np
import sys
import cv2

def target_category_loss(x, category_index, nb_classes):
    return tf.mul(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # Normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def preprocess(img_path):
    original = image.load_img(img_path)
    original = image.img_to_array(original, dim_ordering='tf')
    resized = image.load_img(img_path, target_size=(150, 150))
    preprocessed = image.img_to_array(resized, dim_ordering='tf')
    preprocessed = preprocessed.astype("float") / 255.0  # Normalize the array into range 0...1
    preprocessed = np.expand_dims(preprocessed, axis=0)
    return original, preprocessed

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    # The input model (VGG16) is added to the Sequential model, and appears as "model_1" in the model summary.
    # The layers of this model may be accessed using model.layers[0].layers, as shown below under
    # variable "conv_output".

    nb_classes = 3

    # So the Lambda layer is essentially an one-hot encoded vector whose length corresponds to the number of classes?

    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape=target_category_loss_output_shape))

    # Construct a dictionary of layers in the original model
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0].layers])

    loss = K.sum(model.layers[-1].output)  # Calculate loss?
    conv_output = layer_dict[layer_name].output  # Get output from the second convolutional block
    grads = normalize(K.gradients(loss, conv_output)[0])  # Get gradients and apply L2 normalization
    gradient_function = K.function([model.layers[0].input, K.learning_phase()], [conv_output, grads])  # Define gradient function

    output, grads_val = gradient_function([image, 0])  # Apply gradient function to original image; 0 for K.learning_phase()
    output, grads_val = output[0, :], grads_val[0, :, :, :]  # Remove the dummy vector required to pass the image through the CNN

    weights = np.mean(grads_val, axis=(0, 1))  # Get weights
    cam = np.ones(output.shape[0:2], dtype=np.float32)  # Create a NumPy array to host the CAM

    # Loop over the feature maps and add to CAM
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]  # Multiply each feature map by its weight; add to the CAM array (+=)

    cam = np.maximum(cam, 0)  # ???
    cam /= np.max(cam)  # Divide by maximum value

    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # Apply JET colour map to CAM
    cam = np.float32(cam)  # ???
    cam = 255 * cam / np.max(cam)  # ???

    # Return CAM as NumPy array
    return np.uint8(cam)

# Load and preprocess input
original_input, preprocessed_input = preprocess(sys.argv[1])

# Load model & weights
model = CNNArchitecture.select('MiniVGGNet', 150, 150, 3, 3)
model.load_weights('tankbuster/engine/weights.h5')

# Get accurate predictions
preds = npbust(original_input)
labels = {'other': 0, 't-72': 1, 'bmp': 2}  # Class labels
class_label = max(preds, key=preds.get)
predicted_class = labels[class_label]

# Create CAM
cam = grad_cam(model, preprocessed_input, predicted_class, "maxpooling2d_2")

# Scale CAM to input size and combine with input
cam = cv2.resize(cam, (original_input.shape[1], original_input.shape[0]))  # Resize
cam = cam + original_input  # Combine CAM and original image

# Superimpose class label on the image
cv2.putText(cam, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Write CAM to disk
cv2.imwrite("cam.jpg", cam)
