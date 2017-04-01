import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

source_dir = 'data_raw/training'

# Set up the ImageDataGenerator
generator = ImageDataGenerator(rescale=1./255,
                               rotation_range=10,
                               width_shift_range=0.20,
                               height_shift_range=0.05,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest'
                               )

# Walk through the source directory
for (root, subdirs, files) in os.walk(source_dir):

    # Define allowed image extensions
    ext = ['png', 'jpg', 'jpeg']

    # Loop over the files
    for f in files:

        # Check that the extension is allowed
        if f.split('.')[-1] in ext:
            iext = f.split('.')[-1]  # Get extension for the image in question
            path = os.path.join(root, f)  # Get image path
            label = path.split('/')[-2]  # Get class label

            # Read image
            img = cv2.imread(path)

            # Resize image
            img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)

            # Define target directory
            target_dir = os.path.join('data_rsearch', label)

            # Check that target directory exists
            if not os.path.exists(target_dir):

                # Create directory
                os.makedirs(target_dir)

            # Construe a filename for the resized image
            newf = os.path.join('data_rsearch', label, f)

            # Write image on disk
            cv2.imwrite(newf, img)
