#!/usr/bin/env python
# coding: utf-8

"""
It has pre-defined six classes including buildoing, forest, glacier, mountain, sea and street. 
Beisde of model.fit() with the batch_size=32, we set batch_size=1 for  both model.evaluate()
and model.predict()
"""


import tensorflow as tf
from keras import layers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import datetime
from alexnet import AlexNet


# Set up the GPU growth to avoid the sudden stop of the runtime. 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Give the global constants.Please notify BATCH_SIZE for model.fit() and Batch_Size for 
# model.evaluate() and model.predict(). 
EPOCHS = 50
BATCH_SIZE = 32
Batch_Size = 1
image_width = 227
image_height = 227
channels = 3
num_classes = 6


# Call the alexnet model in alexnet.py. 
model = AlexNet((image_width,image_height,channels), num_classes)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# It will output the AlexNet model after executing the command 
model.summary()


# The dataset inlcude the three directories 
train_dir = '/home/mike/Documents/Six_Classify_AlexNet/seg_train/seg_train'
test_dir = '/home/mike/Documents/Six_Classify_AlexNet/seg_test/seg_test'
predict_dir = '/home/mike/Documents/Six_Classify_AlexNet/seg_pred/'


# keras.preprocessing.image.ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255)

# keras.preprocessing.image.DirectoryIterator
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(image_width,image_height),
                                                    class_mode='categorical')

train_num = train_generator.samples


# Please start the following tensorboard in the Ubuntu Terminal after executing the script. 
# $ Tensorboard --logdir logs/fit
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]


# Set verbose=1 (or verbose=0) for visibale (or invisible) training procedure. 
model.fit(train_generator,
          epochs=EPOCHS,
          steps_per_epoch=train_num//BATCH_SIZE,
          callbacks=callback_list,
          verbose=1)


# It is the test generator as similar as the above. 
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(image_width,image_height), 
                                                  class_mode='categorical')

test_num = test_generator.samples


# Evalute the trained model and return both loss and the test accuracy. 
evals = model.evaluate(test_generator,
                        verbose=1,
                        batch_size=Batch_Size,
                        steps=test_num//Batch_Size)

print("Loss = " + str(evals[0]))
print("Test Accuracy = " + str(evals[1]))


# Predict the classifcation given the implicit steps=7301 for selecting the specific image number. 
predict_datagen = ImageDataGenerator(rescale=1.0/255)

predict_generator = predict_datagen.flow_from_directory(predict_dir, 
                                                        target_size=(image_width,image_height),
                                                        batch_size=Batch_Size,
                                                        class_mode='categorical')

predict_num = predict_generator.samples


# Make the prediction for any one of the predicted images 
predictions = model.predict(predict_generator,
                            verbose=1,
                            batch_size=Batch_Size,
                            steps=predict_num//Batch_Size)


# Plot the discriptive diagram 
imshow(predict_generator[5800][0][0])
plt.imsave("predicted1.png", predict_generator[5800][0][0])

predictions[5800]

print(predictions[5800])