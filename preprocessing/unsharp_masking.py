# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:18:32 2021

@author: admin
"""

from skimage import io
from skimage.filters import unsharp_mask
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import random

images_path = "augmented_more_images/images/"

aug_images_path = "augmented_more_images/images_unsharped/"

images = []


for im in os.listdir(images_path):
    images.append(os.path.join(images_path, im))

images.sort()    
    
i = 0

while i <= len(images):
    number = i  # PIck a number to select an image & mask
    image = images[number]
    #print(image, mask)
    # image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    unsharped_img = unsharp_mask(original_image, radius=20, amount=1)
    
    new_image_path = "%s/unsharped_image%s.png" % (aug_images_path, i)
    io.imsave(new_image_path, unsharped_img)
    
    i += 1

#sharpened = unsharp_mask(image0, radius=1.0, amount=1.0)



# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12, 12))
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(img, cmap='gray')
# ax1.title.set_text('Input Image')
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(unsharped_img, cmap='gray')
# ax2.title.set_text('Unsharped Image')

# plt.show()