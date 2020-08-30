import os
import numpy as np
import matplotlib.pyplot as plt
from six_classify import predictions, predict_generator

train_dir = '/home/mike/Documents/Six_Classify_AlexNet/seg_train/seg_train'

def get_category(predicted_output):
    return os.listdir(train_dir)[np.argmax(predicted_output)] 

print(get_category(predictions[512]))

fig, axs = plt.subplots(2, 3, figsize=(10,10))

axs[0][0].imshow(predict_generator[1002][0][0])
axs[0][0].set_title(get_category(predictions[1002]))

axs[0][1].imshow(predict_generator[22][0][0])
axs[0][1].set_title(get_category(predictions[22]))

axs[0][2].imshow(predict_generator[1300][0][0])
axs[0][2].set_title(get_category(predictions[1300]))

axs[1][0].imshow(predict_generator[3300][0][0])
axs[1][0].set_title(get_category(predictions[3300]))

axs[1][1].imshow(predict_generator[7002][0][0])
axs[1][1].set_title(get_category(predictions[7002]))

axs[1][2].imshow(predict_generator[512][0][0])
axs[1][2].set_title(get_category(predictions[512]))