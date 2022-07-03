"""
Convert all OASIS scans to png images
"""

import os
import numpy, shutil, os, nibabel
import sys, getopt
import argparse
import imageio
import math
from tqdm import tqdm

image_paths = []
for root, dirs, files in os.walk("./T1w"):
    for file in files:
        if file.endswith(".nii.gz"):
            image_paths.append(os.path.join(root,file)) 
            #print(os.path.join(root, file))
            
slice_counter = 0
rotation_angle = 90

for file in tqdm(image_paths):
    image_array = nibabel.load(file).get_data()
    print(len(image_array.shape))
    # set destination folder
    if not os.path.exists('./T1w-png-converted'):
        os.makedirs('./T1w-png-converted')
    
    # For 3D image inputted    
    if len(image_array.shape) == 3:
        nx, ny, nz = image_array.shape
        total_slices = image_array.shape[2]
        # iterate through slices
        #for current_slice in range(0, total_slices):
        for current_slice in range(math.floor(0.35*total_slices), math.floor(0.75*total_slices)):
            # alternate slices
            if (slice_counter % 1) == 0:
                # rotate or no rotate
                if rotation_angle == 90:
                    data = numpy.rot90(image_array[:, :, current_slice])
                elif rotation_angle == 180:
                    data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice]))
                elif rotation_angle == 270:
                    data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice])))
                 #alternate slices and save as png
                if (slice_counter % 1) == 0:
                    #print('Saving image...')
                    image_name = file[6:23]+file[24:29] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
                    imageio.imwrite(image_name, data)
                    #print('Saved.')
                    #move images to folder
                    #print('Moving image...')
                    src = image_name
                    shutil.move(src, './T1w-png-converted')
                    slice_counter += 1
                    #print('Moved.')
                    #print('Finished converting images')
    elif len(image_array.shape) == 4:
        nx, ny, nz, nw = image_array.shape
        total_volumes = image_array.shape[3]
        total_slices = image_array.shape[2]
        for current_volume in range(0, total_volumes):
            # iterate through slices
            #for current_slice in range(0, total_slices):
            for current_slice in range(math.floor(0.35*total_slices), math.floor(0.75*total_slices)):     
                if (slice_counter % 1) == 0:
                    if rotation_angle == 90:
                        data = numpy.rot90(image_array[:, :, current_slice, current_volume])
                    elif rotation_angle == 180:
                        data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume]))
                    elif rotation_angle == 270:
                        data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume])))

                    #alternate slices and save as png
                    if (slice_counter % 1) == 0:
                        #print('Saving image...')
                        image_name = file[6:23] + file[24:29] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
                        imageio.imwrite(image_name, data)
                        #print('Saved.')
                        #move images to folder
                        #print('Moving image...')
                        src = image_name
                        shutil.move(src, './T1w-png-converted')
                        slice_counter += 1
                        #print('Moved.')
                        #print('Finished converting images')
