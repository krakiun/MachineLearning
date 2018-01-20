
# coding: utf-8
import sys
# No need to modify this since you can just supply train as an argument
TRAIN = False

# Set this to True if you want to create the training data
CREATE_DATA = False


if len(sys.argv) > 1:
    if sys.argv[1] == "train":
        TRAIN = True
    else:
        IMG_PATH = sys.argv[1]
else:
    print("You need to supply an argument to the image path/train")
    sys.exit(1)

# We do the imports here in case we do not supply the correct arguments
# we won't wait to import everything

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np
import os, glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set IMG_DIR to the image directory where the training data is
IMG_DIR = "SET_THIS_TO_YOUR_PATH"  # Example: /home/user/DogVsCat/train/
IMG_SIZE = 100


def label_data(data):
    """Input a string(cat/dog) and return a list.
    cat --> [1, 0]
    dog --> [0, 1]
    """
    if   data == 'cat': return [1, 0]
    elif data == 'dog': return [0, 1]


def make_image(file):
    """Create an image that can be fed to the DNN to predict."""
    image = cv2.imread(file, 0)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return np.array([np.array(image), np.array([0, 0])])


def create_training_data():
    """
    Create the training data:
    For every image:
      You transform it and add it to a list ([np.array(image)],[np.array(cat/dog)])
    Shuffle the data
    Save the data so we don't do it every time
    """
    training_data = []
    os.chdir(IMG_DIR)
    for file in tqdm(glob.glob("*.jpg")):
        label = file.split("/")[-1][:3]  # Get the `class` (Either a cat [1,0] or a dog [0,1])
        label = label_data(label)  # From cat/dog to list representation
        image = cv2.imread(file, 0)   # Read the image and make it grayscale
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(image), np.array(label)])
    np.random.shuffle(training_data)
    np.save("train_data.npy", training_data)

if CREATE_DATA:
    create_training_data()

# Load the training data
training_data = np.load(IMG_DIR + "/train_data.npy")


# Create the neural network
dnn = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

for x in range(5):
    dnn = conv_2d(dnn, 32, 2, activation='relu')
    dnn = max_pool_2d(dnn, 2)

    dnn = conv_2d(dnn, 64, 2, activation='relu')
    dnn = max_pool_2d(dnn, 2)


dnn = fully_connected(dnn, 1024, activation='relu')
dnn = dropout(dnn, 0.8)
dnn = fully_connected(dnn, 2, activation='softmax')
dnn = regression(dnn, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')

model = tflearn.DNN(dnn, checkpoint_path="model.tfl.ckpt", max_checkpoints=1)


if TRAIN:
    # Split the data
    middle_split = int(25000 / 4) * 3  # Get 75% for train and 25% for test
    train = training_data[:middle_split]
    test = training_data[middle_split:]

    # Create the input and the labels
    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = [i[1] for i in train]
    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in test]
    # Train the model
    model.fit({'input': X}, {'target': Y}, n_epoch=20, validation_set=({'input': test_x}, {'target': test_y}), show_metric=True)

    # After training, save the model
    print("Saving the model")
    model.save("model.tfl")
    print("Saved the model")

else:
    model.load("model.tfl")
    img = mpimg.imread(IMG_PATH)
    model_image = make_image(IMG_PATH)[0]
    model_image = model_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict(model_image)
    if np.argmax(model_out) == 1:
        str_label = "Dog"
    else:
        str_label = "Cat"

    img_plot = plt.imshow(img)
    plt.title(str_label)
    plt.show()
