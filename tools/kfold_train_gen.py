# Import the necessary packages
import cPickle as pickle
import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from tankbuster.cnn import CNNArchitecture
from keras import backend as K
from sklearn.model_selection import StratifiedKFold

# Set variables for training the model
arch = "ConvNet"  # Architecture
opt = "SGD"  # Optimizer
learning_rate = 0.001  # Learning rate
target_size = (150, 150)  # Target size for data augmentation

# Configure callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50,
                              cooldown=50, min_lr=0.0001, verbose=1)

# Path to pre-trained weights file, if used. Otherwise None.
weights = None

# Configure the validation parameters
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Configure TensorFlow session to allow GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target, config=config)

# Register TensorFlow session with Keras
K.set_session(sess)

# Set up a list for keeping validation scores
scores = []

# Read data & labels
data, labels, classes = [], [], {}

for (root, subdirs, files) in os.walk('data_raw_all'):
    # Assign a numerical identifier to each class directory
    for i, class_dir in enumerate(subdirs):
        classes[class_dir] = i

    # Define allowed image extensions
    ext = ['png', 'jpg', 'jpeg']

    # Loop over the files in each directory
    for f in files:
        if f.split('.')[-1] in ext:  # Check file extension
            path = os.path.join(root, f)  # Get image path
            label = path.split('/')[-2]  # Extract class label from path
            numlabel = classes[label]  # Get numerical label from the dict

            print "*** Now processing {} / {} / {} ...".format(path,
                                                               label,
                                                               numlabel)

            # Load and preprocess image
            image = load_img(path, target_size=target_size)  # Load image
            features = img_to_array(image)  # Convert image to numpy array

            labels.append(numlabel)  # Append label to list
            data.append(features)  # Append features to list

# Convert data and labels to numpy arrays
data = np.asarray(data, dtype=np.float32)
labels = np.asarray(labels, dtype=np.float32)

# Initiate TensorFlow session
with sess.as_default():
    for n, (train, test) in enumerate(kfold.split(data, labels)):

        # Select CNN architecture
        print "Setting up CNN architecture: {} ...".format(arch)
        model = CNNArchitecture.select(arch, target_size[0],
                                       target_size[1], 3, 3)
        if opt == "SGD":
            optimizer = SGD(lr=learning_rate, decay=1e-5,
                            momentum=0.9, nesterov=True)
        if opt == "RMSProp":
            optimizer = RMSprop(lr=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                      metrics=['categorical_accuracy'])

        # Split the data
        traindata, trainlabels = data[train], labels[train]
        testdata, testlabels = data[test], labels[test]

        # Convert integer labels into one-hot encoded vectors
        trainlabels = np_utils.to_categorical(trainlabels, 3)
        testlabels = np_utils.to_categorical(testlabels, 3)

        if weights:
            print "Loading pre-trained weights from {} ...".format(weights)
            model.load_weights(weights)

        with tf.device('/gpu:0'):
            # Set up generator for training data
            training_generator = ImageDataGenerator(rescale=1./255,
                                                    rotation_range=10,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.05,
                                                    shear_range=0.2,
                                                    zoom_range=0.2,
                                                    horizontal_flip=True,
                                                    fill_mode='nearest'
                                                    )

            # Generate training data
            training_data = training_generator.flow(traindata,
                                                    trainlabels,
                                                    batch_size=256
                                                    )

        with tf.device('/gpu:1'):
            # Set up generator for validation data
            validation_generator = ImageDataGenerator(rescale=1./255,
                                                      rotation_range=10,
                                                      width_shift_range=0.2,
                                                      height_shift_range=0.05,
                                                      shear_range=0.2,
                                                      zoom_range=0.2,
                                                      horizontal_flip=True,
                                                      fill_mode='nearest'
                                                      )

            # Generate validation data
            validation_data = validation_generator.flow(testdata,
                                                        testlabels,
                                                        batch_size=32
                                                        )

        training = model.fit_generator(training_data,
                                       samples_per_epoch=2048,
                                       nb_epoch=1000,
                                       validation_data=validation_data,
                                       nb_val_samples=256,
                                       callbacks=[reduce_lr]
                                       )

        (loss, accuracy) = model.evaluate(testdata,
                                          testlabels,
                                          batch_size=32,
                                          verbose=1)

        # Save weights
        model.save_weights('test_output/cv/convnet-cv-fold_%s.h5' % (n + 1),
                           overwrite=True)

        with open('test_output/cv/convnet-history-fold_%s.pkl' % (n + 1), 'wb') \
                as histfile:
            pickle.dump(training.history, histfile)

        # Append accuracy to list of scores
        scores.append(accuracy)

print "%.4f (STDEV %.4f)" % (np.mean(scores), np.std(scores))
print "Best result for fold %s" % np.argmax(scores)

