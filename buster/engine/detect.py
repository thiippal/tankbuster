# Import the necessary packages
import cv2
from buster.cnn import CNNArchitecture

def bust(image):
    # Load and process the image
    original = cv2.imread(image)
    resized = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)  # Resize
    rescaled = resized.astype("float") / 255.0  # Scale into range 0...1
    reshaped = rescaled.reshape((1,) + rescaled.shape)  # Reshape for input to CNN

    # Load the CNN architecture and pre-trained weights, compile the model
    model = CNNArchitecture.select('MiniVGGNet', 256, 256, 3, 2)
    model.load_weights('buster/classifier/weights.hdf5')
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    # Return the prediction
    prediction = model.predict_proba(reshaped, verbose=0)[0]

    # Check the predicted labels
    other = prediction[0]
    tank = prediction[1]

    if tank > 0.10:
        print "[POSITIVE] File {} contains a tank with {:.2f}% confidence.".format(image, tank * 100)
    else:
        print "[NEGATIVE] File {} does not contain a tank with {:.2f}% confidence.".format(image, other * 100)

