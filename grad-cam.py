# from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Lambda
from keras.models import Sequential
from tankbuster.cnn import CNNArchitecture
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
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)  # Remove VGG16 preprocess
    return x

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

    # Set testing phase = no dropout
    K.set_learning_phase(0)

    # Construct a dictionary of layers
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0].layers])

    loss = K.sum(model.layers[-1].output)
    conv_output = layer_dict[layer_name].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input, K.learning_phase()], [conv_output, grads])

    output, grads_val = gradient_function([image, 0])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)

preprocessed_input = load_image(sys.argv[1])

# model = VGG16(weights='imagenet')

model = CNNArchitecture.select('MiniVGGNet', 150, 150, 3, 3)
model.load_weights('tankbuster/engine/tf-weights.h5')

predicted_class = np.argmax(model.predict(preprocessed_input))

cam = grad_cam(model, preprocessed_input, predicted_class, "maxpooling2d_2")
cv2.imwrite("cam.jpg", cam)
