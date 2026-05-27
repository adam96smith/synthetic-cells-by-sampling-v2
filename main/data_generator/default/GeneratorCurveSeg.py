'''

Curvature Segmentation of Synthetic Motile Cells using Masks from GeneratorMask.py

'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import pickle
from tifffile import imread, imwrite
import h5py

import os
import glob
import sys

## add path to my_elektronn (in main)
root_dir = '/'
for s in sys.path[0].split('/')[:-1]:
    if s != '':
        root_dir += s+'/'
(sys.path).append(root_dir)
        
import time
from scipy.ndimage import convolve, binary_dilation
from skimage.morphology import remove_small_objects

import my_elektronn3
from my_elektronn3.custom.curvature import *
import re

import argparse

parser = argparse.ArgumentParser(description='Identify regions of High Curvature.')
parser.add_argument('--input-dir', type=str, default='train_data/', help='Target Directory')
parser.add_argument('--aniso-factor', type=int, default=8, help='Anisotropic Factor.')
parser.add_argument('--size', type=int, default=15, help='Size of Ball to Calculate Curvature Metric')
parser.add_argument('--threshold', type=int, default=.7, help='Metric Threshold to get Curved Regions')
# parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()

assert args.size % 2 == 1 # Size must be odd
assert 0 <= args.threshold <= 1 # Threshold must be between [0,1]

# if not args.disable_cuda and torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
    
# input directory
input_dir = args.input_dir
mask_files = sorted(glob.glob(input_dir+'mask_*.h5'))

# print(mask_files)

generation_times = []
for i, mask_file in enumerate(mask_files):
    start_time = time.time()
    if i % 50 == 0:
        print(f'Curvature Segmentation of Mask {i+1}/{len(mask_files)}...')
        
    # load data
    metadata = {}
    with h5py.File(mask_file,'r') as f:
        dataset = f['data']
        mask = dataset[...]

    # segment the image based on curvature
    curvature_array = curvature_approximation_3d(mask[0], r=args.size, aniso_factor=args.aniso_factor, planar=False)

    # threshold curvature array
    curvature_segmentation = curvature_array > .7
    kern = circleKern_aniso(15, aniso_factor=args.aniso_factor)

    # localise the regions of high positive curvature
    curvature_segmentation = remove_small_objects(curvature_segmentation, 5)
    curvature_segmentation = binary_dilation(curvature_segmentation, kern)
    
    file_index = re.findall(r'\d+', mask_file)[-2]

    with h5py.File(f'{input_dir}curv_seg_{file_index}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=(curvature_segmentation[np.newaxis]).astype(bool))

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