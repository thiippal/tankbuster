# Import the necessary packages
import cPickle as pickle
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tankbuster.cnn import CNNArchitecture
from keras import backend as K
import tensorflow as tf


# Set variables for training the model
arch = "ConvNet"  # Architecture
opt = "SGD"  # Optimizer
learning_rate = 0.01  # Learning rate
target_size = (150, 150)  # Target size for data augmentation

# Path to pre-trained weights file, if used. Otherwise None.
weights = None

# Configure callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50,
                              cooldown=50, min_lr=0.00000001, verbose=1)

# Configure TensorFlow session to allow GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target, config=config)

# Register TensorFlow session with Keras
K.set_session(sess)

# Initiate TensorFlow session
with sess.as_default():

    # Select CNN architecture
    print "Setting up CNN architecture: {} ...".format(arch)
    model = CNNArchitecture.select(arch, target_size[0], target_size[1], 3, 3)
    if opt == "SGD":
        optimizer = SGD(lr=learning_rate, decay=1e-5,
                        momentum=0.9, nesterov=True)
    if opt == "RMSProp":
        optimizer = RMSprop(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    # Print model summary for log
    print model.summary()

    if weights:
        print "Loading pre-trained weights from {} ...".format(weights)
        model.load_weights(weights)

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

    # Generate training data
    training_data = training_generator.flow_from_directory(
        'data_raw_split/training',
        target_size=target_size,
        batch_size=256,
    )

    # Generate validation data
    validation_data = validation_generator.flow_from_directory(
        'data_raw_split/validation',
        target_size=target_size,
        batch_size=32,
    )

    # Start training
    training = model.fit_generator(training_data,
                                   samples_per_epoch=2048,
                                   nb_epoch=2000,
                                   validation_data=validation_data,
                                   nb_val_samples=256,
                                   callbacks=[scheduler]
                                   )

    # Create timestamp for filename
    stamp = str(datetime.now()).split(' ')[0]

    print "*** Saved model, weights and history to file ..."
    model.save_weights('test_output/weights-%s-%s-%s-gen.h5' % (arch,
                                                                stamp,
                                                                learning_rate),
                       overwrite=True)
    with open('test_output/history_%s_%s_%s.pkl' % (arch,
                                                    stamp,
                                                    learning_rate),
              'wb') as hfile:

        # Dump model history dict into file
        pickle.dump(training.history, hfile)
