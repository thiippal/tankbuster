import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout,\
    Flatten, Dense
from keras.regularizers import l2
from sklearn.model_selection import RandomizedSearchCV
from keras import backend as K
from keras.optimizers import SGD


# Set backend and image dimension ordering
backend = K.backend()

if backend == 'theano':
    K.set_image_dim_ordering('th')

print "*** Using {} backend with {} dimension ordering ...".format(
    K.backend(), K.image_dim_ordering())


# Set up a dictionary for class identifiers
classes = {}


# Define a function for preprocessing data
def prepare_data(sourcedir):
    """
    This function parses a directory for images, extracts their labels,
    and converts them into numpy arrays.

    Params:
        sourcedir: The root directory to parse for images.

    Returns:
        Two NumPy arrays containing the images (as normalized NumPy arrays) and
        labels (as one-hot encoded vectors).
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

                numlabel = classes[label]  # Get numerical label from the classes dict
                image = load_img(path)  # Load image
                features = img_to_array(image)  # Convert image to numpy array

                labels.append(numlabel)  # Append label to list
                data.append(features)  # Append features to list

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Convert numerical labels into one-hot encoded vectors
    labels = np_utils.to_categorical(labels, len(classes))

    # Normalize data into range 0...1
    data = data.astype('float') / 255.0

    # Return data and labels
    return data, labels


# Define a function to construct a MiniVGGNet for the scikit-learn wrapper
def create_minivggnet(l2_lambda=0.0, dropout=0.0, learning_rate=0.1, nodes=0):
    # Initialize the model
    kmodel = Sequential()

    # Define the first set of  CONV -> RELU -> CONV -> RELU -> POOL layers
    kmodel.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150),
                             W_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(Convolution2D(32, 3, 3, W_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(MaxPooling2D(pool_size=(2, 2)))

    # Define the second set of CONV -> RELU -> CONV -> RELU -> POOL layers
    kmodel.add(Convolution2D(64, 3, 3, W_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(Convolution2D(64, 3, 3, W_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps
    kmodel.add(Flatten())

    # Add dropout
    kmodel.add(Dropout(dropout))

    # Add a dense layer followed by an activation
    kmodel.add(Dense(nodes, W_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))

    # Add dropout
    kmodel.add(Dropout(dropout))

    # Define the Softmax classifier
    kmodel.add(Dense(3))
    kmodel.add(Activation("softmax"))

    # Define the optimizer
    sgd = SGD(lr=learning_rate)

    # Compile model
    kmodel.compile(loss="categorical_crossentropy", optimizer=sgd,
                   metrics=['categorical_accuracy'])

    return kmodel

# Prepare data
data, labels = prepare_data('data_rsearch_150/')

# Define lists of parameters to be evaluated
batch_size = [32, 64, 128, 256]
l2_lambda = [0.01, 0.001, 0.0001]
dropout = [0.25, 0.5, 0.75]
nodes = [64, 128, 256, 512]
learning_rate = [0.1, 0.01, 0.001, 0.0001]

# Construct the parameter dictionary
param_dict = dict(batch_size=batch_size, l2_lambda=l2_lambda, dropout=dropout,
                  nodes=nodes, learning_rate=learning_rate)

# Build Keras model
model = KerasClassifier(build_fn=create_minivggnet, verbose=1, nb_epoch=50)

# Configure random search
rsearch = RandomizedSearchCV(estimator=model,
                             n_iter=50,
                             param_distributions=param_dict,
                             random_state=42,
                             scoring='neg_log_loss',
                             verbose=2,
                             cv=3
                             )

# Perform random search
rsearch.fit(data, labels)

# Print best result
print "Best result: %f using %s" % (rsearch.best_score_,
                                    rsearch.best_params_)

# Assign results to lists
mean = rsearch.cv_results_['mean_test_score']
std = rsearch.cv_results_['std_test_score']
param = rsearch.cv_results_['params']

# Loop over the lists and print the result:
if __name__ == '__main__':
    for m, s, p in zip(mean, std, param):
        print "Mean: %f (SD: %f) with %r" % (m, s, p)
