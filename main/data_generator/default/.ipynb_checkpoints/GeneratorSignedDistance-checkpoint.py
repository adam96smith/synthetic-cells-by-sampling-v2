'''

Curvature Segmentation of Synthetic Motile Cells using Masks from GeneratorMask.py

'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import pickle
from tifffile import imread, imwrite

import os
import glob
import sys

## add path to my_elektronn (in main)
if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())
    
import time
from scipy.ndimage import convolve, binary_erosion, distance_transform_edt
from skimage.morphology import remove_small_objects

# import my_elektronn3
# from my_elektronn3.custom.curvature import *
from utils import load_config
import re

import argparse

parser = argparse.ArgumentParser(description='Identify regions of High Curvature.')
parser.add_argument('--input-dir', type=str, default='train_data/', help='Target Directory')
parser.add_argument('--config', required=True, help='Global Config. for Sampling.')
parser.add_argument('--global-config', required=True, help='Global Config. for Sampling.')
args = parser.parse_args()


# input directory
input_dir = args.input_dir
mask_files = sorted(glob.glob(input_dir+'mask_*.tif'))

config = load_config(args.config)
sf = config['SHAPE']['DOWNSAMPLE']

global_config = load_config(args.global_config)
sampling = global_config['SAMPLING']

# scale sampling
sampling = (sampling[0], sf*sampling[1], sf*sampling[2])

# print(mask_files)

generation_times = []
for i, mask_file in enumerate(mask_files):
    start_time = time.time()
    if i % 50 == 0:
        print(f'Signed Distance Functions for Mask {i+1}/{len(mask_files)}...')
        
    # load data
    instances = imread(mask_file)

    # Signed Distance Transform
    SDF = np.zeros(instances.shape, np.float32)
    for v in np.unique(instances):
        if v == 0:
            SDF -= distance_transform_edt((instances==v), sampling=sampling)
        else:
            SDF += distance_transform_edt(binary_erosion(instances==v), sampling=sampling) # think this is correct
    
    file_index = re.findall(r'\d+', mask_file)[-1]

    os.makedirs(f'{input_dir}SDF/', exist_ok=True)
    imwrite(f'{input_dir}SDF/dist_map_{file_index}.tif', SDF)

    # Calculate time taken for this iteration 
    iteration_time = time.time() - start_time 
    generation_times.append(iteration_time) 
    
    if i % 50 == 0:
        # Calculate the average time taken so far 
        avg_time_per_iteration = sum(generation_times) / len(generation_times) 
        
        # Estimate remaining time 
        remaining_iterations = len(mask_files) - (i + 1) 
        estimated_time_left = avg_time_per_iteration * remaining_iterations 
        
        # Convert estimated time to hours and minutes 
        hours_left = int(estimated_time_left // 3600) 
        minutes_left = int((estimated_time_left % 3600) // 60) 
        seconds_left = int(estimated_time_left % 60) 
        
        # Output estimated time remaining 
        print(f"{i+1}/{len(mask_files)}, Average Time per Cell: {avg_time_per_iteration}s, Estimated time left - {hours_left}h {minutes_left}m {seconds_left}s")