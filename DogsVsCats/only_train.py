# coding: utf-8

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
import numpy as np
import os, glob # To get the images
import cv2
from tqdm import tqdm

# Set IMG_DIR to the image directory where the training data is
IMG_DIR = "SET_THIS_TO_YOUR_PATH"
IMG_SIZE = 100


def label_data(data):
    """Input a string(cat/dog) and return a list.
    cat --> [1, 0]
    dog --> [0, 1]
    """
    if   data == 'cat': return [1, 0]
    elif data == 'dog': return [0, 1]


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
        image = cv2.imread(file, 0)  # Read the image and make it grayscale
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(image), np.array(label)])
    np.random.shuffle(training_data)
    np.save("train_data.npy", training_data)

#create_training_data()


training_data = np.load(IMG_DIR + "/train_data.npy")


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

middle_split = int(25000 / 4) * 3  # Get 75% for train and 25% for test
train = training_data[:middle_split]
test = training_data[middle_split:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'target': Y}, n_epoch=20, validation_set=({'input': test_x}, {'target': test_y}), show_metric=True)


print("Saving the model")

model.save("model.tfl")

print("Saved the model")
