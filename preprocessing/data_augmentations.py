import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
import albumentations as A
import tensorflow as tf

images_to_generate = 10000

images_path = "full_size_data/images"
masks_path = "full_size_data/masks"
aug_images_path = "augmented_more_images/images/"
aug_masks_path = "augmented_more_images/masks/"
images = []
masks = []

for im in os.listdir(images_path):
    images.append(os.path.join(images_path, im))

for msk in os.listdir(masks_path):
    masks.append(os.path.join(masks_path, msk))
    
images.sort()
masks.sort()

aug = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.4),
    A.HorizontalFlip(p=0.6),
    A.Resize(256,256, p=1),
])




i=0

while i <= images_to_generate:
    number = random.randint(0, len(images) - 1)  # PIck a number to select an image & mask
    image = images[number]
    mask = masks[number]
    print(image, mask)
    # image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    original_mask = io.imread(mask)

    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

    new_image_path = "%s/augmented_image%s.png" % (aug_images_path, i)
    new_mask_path = "%s/augmented_mask%s.png" % (aug_masks_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    i = i + 1



