
# coding: utf-8
import sys


if len(sys.argv) > 1:
    IMG_PATH = sys.argv[1]
else:
    print("You need to supply an argument to the image path")
    sys.exit(1)

# We do the imports here in case we do not supply the correct arguments
# we won't wait to import everything

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMG_SIZE = 100


def make_image(file):
    image = cv2.imread(file, 0)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return np.array([np.array(image), np.array([0, 0])])


# Define the model
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

model = tflearn.DNN(dnn, checkpoint_path="model.tfl.ckpt")

# Load the model
model.load("model.tfl")
img = mpimg.imread(IMG_PATH)
model_image = make_image(IMG_PATH)[0]
model_image = model_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)

model_out = model.predict(model_image)
if np.argmax(model_out) == 1:
    str_label = "Dog"
else:
    str_label = "Cat"

imgplot = plt.imshow(img)
plt.title(str_label)
plt.show()
