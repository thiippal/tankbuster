import cPickle as pickle
import h5py
from datetime import datetime
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2


# Set up variables
learning_rate = 0.01

# Load features and labels from file
with h5py.File('data_features/resnet50_features-mar17.h5', 'r') as hf:
    data = hf["features"][:]
    labels = hf["labels"][:]

# Compile the model
model = Sequential()
model.add(Flatten(input_shape=data.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
optimizer = SGD(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

training = model.fit(data, labels, validation_split=0.1, batch_size=32,
                     shuffle=True, nb_epoch=250, verbose=1)

# Create timestamp for filename
stamp = str(datetime.now()).split(' ')[0]

print "*** Saved weights and history to file ..."
model.save_weights('test_output/weights-%s-%s-tl.h5' % (stamp, learning_rate),
                   overwrite=True)
with open('test_output/history_%s_%s-tl-gen.pkl' % (stamp, learning_rate),
          'wb') as hfile:
    pickle.dump(training.history, hfile)  # Dump model history dict into file
