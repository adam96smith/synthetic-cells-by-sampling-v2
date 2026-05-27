"""
Test trained Model on real data

Use config files found in config/ to set parameters a model.

The script runs the models and saves the outputs in _X_RES or _RES folders. 

No evaluation is performed.

"""

import argparse
import logging
import h5py
import glob
import os
import random
import time
import re
import sys
import yaml
from tqdm import tqdm
from pathlib import Path
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch import nn
from torch import optim
import numpy as np
import math
import pickle
from tifffile import imread, imwrite

# Set up all RNG seeds, set level of determinism
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


## Root for Custom Elektronn3
if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

# Don't move this stuff, it needs to be run this early to work
import my_elektronn3
my_elektronn3.select_mpl_backend('Agg')
# logger = logging.getLogger('elektronn3log')
# Write the flags passed to python via argument passer to logfile
# They will appear as "Namespace(arg1=val1, arg2=val2, ...)" at the top of the logfile
# logger.debug("Arguments given to python via flags: {}".format(args))

# ## Original
from my_elektronn3.data import PatchCreator, transforms, utils, get_preview_batch
from my_elektronn3.training import Trainer, Backup, metrics
from my_elektronn3.training import SWA
from my_elektronn3.modules import CombinedLoss, CurvatureWeightedLoss
from my_elektronn3.custom.image_filtering import MedianFiltering, GaussianFiltering, ConvZFiltering

# from boundary_loss import BoundaryLoss # from https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py
from my_elektronn3.models.unet import UNet

from scipy.ndimage import median_filter, maximum_filter, gaussian_filter, zoom
from skimage import exposure
from utils import load_config, upscale_by_slice, custom_post_process, nearest_power_of_two, dataset_properties

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
parser.add_argument('--model-version', type=str, default='model_best.pt',
                    help='Model Name.')
parser.add_argument('--threshold', default=None, 
                    help='Theshold for Model Output Probabilities')
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
sampling = global_params['SAMPLING']
aniso_factor = nearest_power_of_two(sampling[0]/sampling[1])

TEST_SCALE = config['evaluation']['EVAL_SCALE']

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')    
    
## Load Model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(f'{args.model_dir}{args.model_name}/{args.model_version}', map_location=device, weights_only=False)
model.eval()

### Evaluate model on real training data

image_files = sorted(glob.glob(f'{args.data_root}{args.test_dir}[0-9][0-9]/*.tif'))

# if args.final: # Run for all images
    
#     image_files = sorted(glob.glob(f'{args.data_root}{args.test_dir}[0-9][0-9]/*.tif'))
    
# else: # find all images with labels
#     labelled_files = sorted(glob.glob(f'{args.data_root}{args.test_dir}[0-9][0-9]_GT/SEG/*.tif'))

#     image_files = []
#     for f in labelled_files:
#         if len(re.findall(r'\d+', f.split('/')[-1])) == 2: # slice labels
#             cell_id, t_lab, _ = re.findall(r'\d+', f)[-3:]
#         elif len(re.findall(r'\d+', f.split('/')[-1])) == 1: # volume labels
#             cell_id, t_lab = re.findall(r'\d+', f)[-2:]
#         else:
#             raise Exception('Error in GT files.')

#         image_files.append(f'{args.data_root}{args.test_dir}{cell_id}/t{t_lab}.tif')

if args.threshold is None:
    # try load optim. else use 0.5
    try:
        with open(f'{args.model_dir}{args.model_name}/model_optim.pkl', 'rb') as f:
            threshold_value = pickle.load(f)['peak_tv']
    except:
        threshold_value = 0.5

else:
    threshold_value = float(args.threshold)
    

for i in tqdm(range(len(image_files)), desc=f'Segmenting Test Image with Threshold {threshold_value:.2f}'):
    image_file = image_files[i]
    
    # Load data
    image = (imread(image_file)[np.newaxis]).astype(np.float32)
    _,_,x_targ,y_targ = image.shape    

    cell_id, t_lab = re.findall(r'\d+', image_file)[-2:]

    T1 = transforms.Normalize(mean=(image.mean(),), std=(image.std(),))

    if TEST_SCALE > 1:
        image = image[:,:,::TEST_SCALE,::TEST_SCALE].astype(np.float32)            
    image, _ = T1(image, [])

    
    # Run model
    inp = torch.from_numpy(image[np.newaxis]).float().to(device)
    with torch.no_grad():  # Save memory by disabling gradients
        output = model(inp).cpu()
    probs = output.softmax(1).detach().numpy()

    if TEST_SCALE > 1:
        prediction = upscale_by_slice(probs[0,1], 
                                      sf=TEST_SCALE, 
                                      xy_target=[x_targ,y_targ], 
                                      order=3,
                                     )[np.newaxis]
    else:
        prediction = probs[:,1]
        
    prediction = custom_post_process(prediction>threshold_value, mode=args.dataset_id)[0]

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
        
    

                        