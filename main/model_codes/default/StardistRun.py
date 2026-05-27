"""
Test trained Model on real data

Use config files found in config/ to set parameters a model.

The script runs the models and saves the outputs in _X_RES or _RES folders. 

No evaluation is performed.

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import re
from tqdm import tqdm
import numpy as np

import glob
from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import Config3D, StarDist3D

np.random.seed(6)
lbl_cmap = random_label_cmap()

from csbdeep.utils.tf import limit_gpu_memory
# adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
limit_gpu_memory(0.8, total_memory=8000)

## Root for Custom Elektronn3
if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

from scipy.ndimage import median_filter, maximum_filter, gaussian_filter, zoom
from skimage import exposure
from utils import load_config, upscale_by_slice, custom_post_process

import argparse

parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--disable-cuda', action='store_true', 
                    help='If True, train on CPU')
parser.add_argument('--data-root', type=str, required=True,
                    help='Path to the base directory containing all datasets.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--test-dir', type=str, required=True,
                    help='Directory with Training Data in default structure.')
parser.add_argument('--model-dir', type=str, required=True,
                    help='Directory to Save Trained Model.')
parser.add_argument('--model-name', type=str, required=True,
                    help='Model Name.')
parser.add_argument('--final', action='store_true', 
                    help='If True, save for CTC')
parser.add_argument('--config', type=str, required=True,
                    help='YAML Config. File')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()


config = load_config(args.config)
if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
aniso_factor = global_params['ANISOTROPIC_FACTOR']
TEST_SCALE = config['evaluation']['EVAL_SCALE'] 
    
## Load Model
model = StarDist3D(None, name=args.model_name, basedir=args.model_dir)
model.load_weights('eval_best.h5')

### Evaluate model on real training data
if args.final: # Run for all images
    
    image_files = sorted(glob.glob(f'{args.data_root}{args.test_dir}[0-9][0-9]/*.tif'))
    
else: # find all images with labels
    labelled_files = sorted(glob.glob(f'{args.data_root}{args.test_dir}[0-9][0-9]_GT/SEG/*.tif'))

    image_files = []
    for f in labelled_files:
        if len(re.findall(r'\d+', f.split('/')[-1])) == 2: # slice labels
            cell_id, t_lab, _ = re.findall(r'\d+', f)[-3:]
        elif len(re.findall(r'\d+', f.split('/')[-1])) == 1: # volume labels
            cell_id, t_lab = re.findall(r'\d+', f)[-2:]
        else:
            raise Exception('Error in GT files.')

        image_files.append(f'{args.data_root}{args.test_dir}{cell_id}/t{t_lab}.tif')
            
for i in tqdm(range(len(image_files)), desc='Evaluating Test Image'):

    image_file = image_files[i]
    
    # Load data
    image = imread(image_file).astype(np.float32)
    _,x_targ,y_targ = image.shape    

    cell_id, t_lab = re.findall(r'\d+', image_file)[-2:]

    image = normalize(image, 3,99.7, axis=(0,1,2))

    image = image[:, ::TEST_SCALE, ::TEST_SCALE].astype(np.float32)
    
    # Run model
    prediction = model.predict_instances(image, n_tiles=model._guess_n_tiles(image), 
                                     show_tile_progress=False)[0]

    if TEST_SCALE > 1:
        prediction = upscale_by_slice(prediction.astype(np.uint8), 
                                      sf=TEST_SCALE, 
                                      xy_target=[x_targ,y_targ]
                                     )
        
    # prediction = custom_post_process(prediction[np.newaxis], mode=args.dataset_id)[0]
    prediction = custom_post_process(prediction[np.newaxis], mode='default')[0]

    

    # Save File
    if args.final:
        sub_folder = f'{cell_id}_RES/'
        
        os.makedirs(f'{args.data_root}{args.test_dir}{sub_folder}', exist_ok=True)

        # save format for CTC
        imwrite(f'{args.data_root}{args.test_dir}{sub_folder}mask{t_lab}.tif', prediction.astype(np.uint16))
        
    else:
        sub_folder = f'{cell_id}_{args.model_name}_RES/'

        os.makedirs(f'{args.data_root}{args.test_dir}{sub_folder}', exist_ok=True)
        
        # save format to save storage
        imwrite(f'{args.data_root}{args.test_dir}{sub_folder}mask{t_lab}.tif', prediction.astype(np.uint16))
        
    

                        