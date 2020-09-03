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
from keras import optimizers
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
BATCH_SIZE = 64
image_width = 150
image_height = 150
channels = 3
num_classes = 1


# Call the cnn/alexnet model 
model = AlexNet((image_width,image_height,channels), num_classes)

# Model configuration
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

# Summary
model.summary


# Directories for the datasets 
train_dir = '/home/mike/Documents/image_gesture/dset_data/train'
val_dir   = '/home/mike/Documents/image_gesture/dset_data/validation'
test_dir   = '/home/mike/Documents/image_gesture/dset_data/train'


# Preprocess the images
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(image_width,image_height),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')
train_num = train_generator.samples

validation_generator = train_datagen.flow_from_directory(val_dir,
                                                         target_size=(image_width,image_height),
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary')
val_num = validation_generator.samples


test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(test_dir,
                                                   target_size=(image_width,image_height),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='binary')
test_num = test_generator.samples


# Get the batch shape
for data_batch, label_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", label_batch)
    
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
# -model.save('/home/mike/Documents/image_gesture/leapGestRecog_small_1.h5')
# -model.save('/home/mike/Documents/image_gesture/leapGestRecog_small_2.h5')


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


# Select the saved model 
model = load_model('/home/mike/Documents/image_gesture/leapGestRecog_small_2.h5')
model.summary()


# Preprocess the image into a 4D tensor
img_path = '/home/mike/Documents/image_gesture/dset_data/test/1/frame_00_03_0001.png'
img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

# The model was trained on the inputs that were preprocessed as follows. 
img_tensor /= 255.

# The shape is (1, 150, 150, 3)
print(img_tensor.shape)


plt.imshow(img_tensor[0])
plt.show()


layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation=activations[0]
print('The 1st layer network size：',first_layer_activation.shape)

# The third channel
plt.matshow(first_layer_activation[0,:,:,3],cmap="viridis")
plt.show()

# The tenth channel
plt.matshow(first_layer_activation[0,:,:,30],cmap="viridis")
plt.show()


# Release the GPU memory
cuda.select_device(0)
cuda.close()