#!/usr/bin/env python
# coding: utf-8

# The model is to train the guesture of images with the Leap Motion 
# T. Mantecón, C.R. del Blanco, F. Jaureguizar, N. García, “Hand Gesture Recognition using Infrared 
# Imagery Provided by Leap Motion Controller”, Int. Conf. on Advanced Concepts for Intelligent Vision
# Systems, ACIVS 2016, Lecce, Italy, pp. 47-57, 24-27 Oct. 2016. (doi: 10.1007/978-3-319-48680-2_5)

# Please download the leapgestrecog dataset from Kaggle. 
# https://www.kaggle.com/gti-upm/leapgestrecog


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# -from keras import optimizers
import matplotlib.pyplot as plt
from alexnet import AlexNet
from keras.models import load_model

from keras import models
from keras.preprocessing import image
import numpy as np
import datetime
from numba import cuda


# Set up the GPU growth to avoid the sudden runtime error
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Assign the global arguments 
EPOCHS = 32
BATCH_SIZE = 100
image_width = 227
image_height = 227
channels = 3
num_classes = 10


# Call the cnn/alexnet model 
model = AlexNet((image_width,image_height,channels), num_classes)

# Model configuration
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Summary
model.summary()


# Directories for the datasets 
train_dir = '/home/mike/Documents/image_gesture/dset_data/train'
val_dir   = '/home/mike/Documents/image_gesture/dset_data/validation'
test_dir  = '/home/mike/Documents/image_gesture/dset_data/test'


# Preprocess the images
train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(image_width,image_height),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

train_num = train_generator.samples

test_datagen = ImageDataGenerator(rescale=1.0/255)

val_generator = test_datagen.flow_from_directory(val_dir,
                                                 target_size=(image_width,image_height),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical')

val_num = val_generator.samples


# Get the batch shape
for data_batch, label_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("label batch shape:", label_batch)
    
    break


# Set the Tensorbaord 
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]


# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_num//BATCH_SIZE,
                    epochs=EPOCHS, 
                    validation_data=validation_generator,
                    validation_steps=val_num//BATCH_SIZE)


# Save the model 
# -model.save('/home/mike/Documents/image_gesture/leapGestRecog_small_categorical.h5')


# Evaluate the model with visulizing the result 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'b', label='Training acc', color='green')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'bo', label='Training loss', color='green')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Release the GPU memory
cuda.select_device(0)
cuda.close()
