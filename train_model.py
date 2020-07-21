from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from imutils import paths
import pickle
import numpy as np
import cv2
import os

LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

data = []
labels = []

for (i, image_file) in enumerate(paths.list_images(LETTER_IMAGES_FOLDER)):
    
    print("[INFO] adding data and labels: {}".format(i + 1))
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20, 20), interpolation = cv2.INTER_AREA)
    image = np.expand_dims(image, axis=2)
    
    label = image_file.split(os.path.sep)[-2]
    
    data.append(image)
    labels.append(label)


data = np.array(data, dtype="float")/255.0
labels = np.array(labels)


(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.2)

lb = LabelBinarizer().fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

# Dumping model labels into pickle file
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Building model
model = Sequential()

model.add(Conv2D(20, (5, 5), input_shape = (20, 20, 1), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Conv2D(50, (5, 5), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

model.add(Flatten())
model.add(Dense(500, activation = "relu"))

# Last layer with 32 nodes for each of our letters
model.add(Dense(32, activation = "softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 32, epochs = 15, verbose = 1)

model.save(MODEL_FILENAME)


