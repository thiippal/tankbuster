import cPickle as pickle
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold

# Set up variables
learning_rate = 0.01

# Load features and labels from file
with h5py.File('data_features/resnet50_features-mar17-cv.h5', 'r') as hf:
    data = hf["features"][:]
    labels = hf["labels"][:]

# Configure the validation parameters
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Set up a list for keeping scores
scores = []

# Loop over the folds
for n, (train, test) in enumerate(kfold.split(data, labels)):
    # Set up and compile the model
    model = Sequential()
    model.add(Flatten(input_shape=data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=1e-5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    # Split the data
    traindata, trainlabels = data[train], labels[train]
    testdata, testlabels = data[test], labels[test]

    # Convert integer labels into one-hot encoded vectors
    trainlabels = np_utils.to_categorical(trainlabels, 3)
    testlabels = np_utils.to_categorical(testlabels, 3)

    # Train the model
    training = model.fit(traindata, trainlabels, batch_size=32, shuffle=True,
                         nb_epoch=75, verbose=1,
                         validation_data=(testdata, testlabels))

    # Evaluate the model
    (loss, accuracy) = model.evaluate(testdata, testlabels, batch_size=32,
                                      verbose=1)

    # Save weights
    model.save_weights('test_output/cv2/topnet-cv-fold_%s.h5' % (n+1),
                       overwrite=True)

    with open('test_output/cv2/topnet-history-fold_%s.pkl' % (n+1), 'wb')\
            as histfile:
        pickle.dump(training.history, histfile)

    # Append accuracy to list of scores
    scores.append(accuracy)

print "%.4f (STDEV %.4f" % (np.mean(scores), np.std(scores))
print "Best result during epoch %s" % np.argmax(scores)
