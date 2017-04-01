import h5py
import numpy as np
import os
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils


# Set up a dictionary for class identifiers
classes = {}

# Initialize ResNet50 with pre-trained weights to be used as feature extractor
model = ResNet50(include_top=False, weights='imagenet',
                 input_tensor=Input(shape=(224, 224, 3)))

# Compile the model and get 2048-d vector from the average pool layer
model = Model(input=model.input,
              output=model.get_layer('avg_pool').output)


# Define a function for extracting features
def extract_features(sourcedir):
    """
    This function parses a directory for images, extracts their labels, applies
    ImageNet preprocessing and extracts CNN codes from ResNet50.

    Params:
        sourcedir: The root directory to parse for images.

    Returns:
        Feature vectors and labels (as one-hot encoded vectors).
    """
    data, labels = [], []

    for (root, subdirs, files) in os.walk(sourcedir):
        # Assign a numerical identifier to each class directory
        for i, class_dir in enumerate(subdirs):
            classes[class_dir] = i

        ext = ['png', 'jpg', 'jpeg']  # Define allowed image extensions

        # Loop over the files in each directory
        for f in files:
            if f.split('.')[-1] in ext:  # Check file extension
                path = os.path.join(root, f)  # Get image path
                label = path.split('/')[-2]  # Extract class label from path
                numlabel = classes[label]  # Get numerical label from the dict

                print "*** Now processing {} / {} / {} ...".format(
                    path, label, numlabel
                )

                # Load and preprocess image
                image = load_img(path, target_size=(224, 224))  # Load image
                features = img_to_array(image)  # Convert image to numpy array

                # Expand image matrix dimensions for input
                features = np.expand_dims(features, axis=0)

                # Apply ImageNet preprocessing
                features = preprocess_input(features, dim_ordering='tf')

                # Extract features
                features = model.predict(features, batch_size=1, verbose=0)

                labels.append(numlabel)  # Append label to list
                data.append(features)  # Append features to list

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Convert numerical labels into one-hot encoded vectors
    # labels = np_utils.to_categorical(labels, len(classes))

    # Return data and labels
    return data, labels

# Extract features
features, labels = extract_features('data_raw_all/')

# Open h5py file
with h5py.File('05_cnn_codes/resnet50_features_cv.h5', 'w') as hf:
    hf.create_dataset("features", data=features)
    hf.create_dataset("labels", data=labels)

