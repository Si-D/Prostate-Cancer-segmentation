# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 18:24:37 2021

@author: admin
"""

import os
from os import path
import shutil

Source_Path = 'S:/Projects/VIT/Karthik Sir-Image Processing/Scripts/Dataset/masks'
Destination = 'S:/Projects/VIT/Karthik Sir-Image Processing/Scripts/full_size_data/masks'
#dst_folder = os.mkdir(Destination)


def main():
    for count, filename in enumerate(os.listdir(Source_Path)):
        dst =  "mask" + str(count) + ".jpg"

        # rename all the files
        os.rename(os.path.join(Source_Path, filename),  os.path.join(Destination, dst))


# Driver Code
if __name__ == '__main__':
    main()