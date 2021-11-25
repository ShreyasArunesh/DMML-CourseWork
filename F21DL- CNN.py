import pandas as pd
import numpy as np
# import seaborn as sn
import matplotlib as plt
import PIL
from PIL import Image
from matplotlib import image
import os
import shutil
from matplotlib import image
from matplotlib import pyplot
import random
import shutil
from pathlib import Path
import glob
import shutil
import os
from pathlib import Path

random.seed(1)
datadir = 'archive/train/'

metadataset = pd.read_csv("archive/train_annotations.csv")
metadataset = metadataset.sample(frac=1, random_state=1).reset_index(drop=True)

# Dividing the image data into sub-folders of classes- Test / train and with 25% of test data
for i in range(metadataset.shape[0]):
    if i < metadataset.shape[0] * 0.75:
        category = "train"
    else:
        category = "test"
    if metadataset.iloc[i,5] == 0:
        target = "Negative"
    else:
        target = "Positive"
    dst_dir = f'{os.getcwd()}/Image_Data/{category}/{target}/'
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(f'{os.getcwd()}/archive/train/{metadataset.iloc[i,9]}', dst_dir)


# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
import keras as ks
from keras.preprocessing.image import ImageDataGenerator


# Part 1 - Data Preprocessing

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[50, 50, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='tan'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit_generator(training_set,
                  steps_per_epoch = 334,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 334)