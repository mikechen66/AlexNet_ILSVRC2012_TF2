#!/usr/bin/env python
# coding: utf-8

# resize_images.py
"""
The data processing script is instrumental to label, convert, resize 
and save the numbers of the images after the data moving script. The 
two directories are one-to-one correspondence. Therefore, we adopt 
the wo-level iterations with the for statments. We resize  all the 
images from 640x240 to 64x64. It is an optional helper script to 
assist in using the smaller size usage. 
"""

import os
from PIL import Image

# Designate both the source and destination directory. 
src_dir = '/home/mike/Documents/image_gesture/leapgestrecog/src_data'
dst_dir = '/home/mike/Documents/image_gesture/leapgestrecog/dst_data'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    
    
# Set classes as a list object and number (of the image) as 0
classes = []
number = 0


# Please notice that item is any element of classes
item = str(classes)
for item in os.listdir(src_dir):
    classes.append(item)


# Define the function of resize()
def resize(src_dir, dst_dir, classes):
    #　Initialize the counter and the current i is different from the above-written i
    i = 0

    # Iterate both index and item with index starting from 1. 
    for index, item in enumerate(classes, 1):
        # Scane images in the source 
        srcit_dir = os.path.join(src_dir, item)
        # Create the dstit_dir with the join function
        dstit_dir = os.path.join(dst_dir, item)  
        #　Judge whether there is a folder
        folder = os.path.exists(dstit_dir)

        if not folder :
            os.makedirs(dstit_dir)
            print(dstit_dir, 'new file')
        else:
            print('There is a current file')
            
        # Please notify imgname is image name
        for imgname in os.listdir(srcit_dir): 
            i += 1
            srcimg_path = os.path.join(srcit_dir, imgname)
            img_data = Image.open(srcimg_path).convert('RGB')
            img_data = img_data.resize((64, 64))
            dstimg_path = os.path.join(dstit_dir, imgname)
            img_data.save(dstimg_path)

            number = i
            
    print('Total images ：%d' % number)


if __name__ == '__main__':
	
    resize(src_dir, dst_dir, classes)
