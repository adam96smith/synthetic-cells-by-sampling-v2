"""
Model Test.

Use config files found in config/ to set parameters a model. 

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
from pathlib import Path
import warnings

root_dir = '/'
for s in os.getcwd().split('/')[:-1]:
    if s != '':
        root_dir += s+'/'
(sys.path).append(root_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch import nn
from torch import optim
import numpy as np
import math
import pickle
from tifffile import imwrite

# Set up all RNG seeds, set level of determinism
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

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

from scipy.ndimage import median_filter, maximum_filter, gaussian_filter
from model_train import custom_post_process, fill_holes_by_slice

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--config', type=str, default='config/example.yaml', help='YAML Config. File')
args = parser.parse_args()

config = load_config(args.config)

## Params
DISABLE_CUDA = config['DISABLE_CUDA']

# INPUTS
MODEL_DIR = config['MODEL_DIR']
MODEL_NAME = config['MODEL_NAME']
TEST_DIR = config['TEST_DIR']


if not DISABLE_CUDA and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')    
    
## Load Model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(f'{MODEL_DIR}{MODEL_NAME}model_best.pt', map_location=device)
model.eval()

# Evaluate the images with labels
   
man_seg_files = sorted([re.findall(r'\d+',s)[-2] for s in glob.glob(f'{root_dir}{EVAL_DIR}*/*_GT/SEG/man_seg**.tif')])

results = {};
all_scores = []
print(f'Evaluating Model on {root_dir}{EVAL_DIR}images ...')
for i, man_seg_file in enumerate(man_seg_files):
    
    if i%1==0:
        print(f'Processing Image {i+1}/{len(test_ids)}...')

    ## if man_segXXX.tif - volume, if man_segXXX_XXX.tif - slice
    N_vals = len(re.findall(r'\d+', man_seg_file.split('/')[-1]))

    if N_vals == 1: # Volume label

        cell_id, t_lab = re.findall(r'\d+', man_seg_file)[-2:]
    
        # Load data
        image = (imread(f'{root_dir}{EVAL_DIR}{cell_id}/t{t_lab}.tif')[np.newaxis]).astype(np.float32)
        mask = (imread(f'{root_dir}{EVAL_DIR}{cell_id}_SEG/SEG/man_seg{t_lab}.tif')[np.newaxis] > .5) # volume data

        # Normalise Images
        T1 = transforms.Normalize(mean=(image.mean(),),
                                  std=(image.std(),))            
        image.astype(np.float32)            
        image, _ = T1(image, [])
        
        # Run model
        inp = torch.from_numpy(image[np.newaxis]).float().to(device)
        with torch.no_grad():  # Save memory by disabling gradients
            output = trainer.model(inp).cpu()
        probs = output.softmax(1).detach().numpy()
        
        prediction = np.array(np.argmax(probs, axis=1)) > 0.5
        prediction = custom_post_process(prediction)

        # Jaccard Scores
        union = np.maximum(prediction, mask)
        intersect = np.zeros_like(mask); intersect[(prediction==1)&(mask==1)] = 1
        
        objective = np.sum(intersect)/np.sum(union)

        try:
            results[cell_id][t_lab] = objective
        except:
            results[cell_id] = {} 
            results[cell_id][t_lab] = objective

        all_scores.append(objective)

    elif N_vals == 2: # Slice label

        cell_id, t_lab, zs_lab = re.findall(r'\d+', man_seg_file)[-3:]
        zs = int(zs_lab[:2].lstrip('0') + zs_lab[-1]) # z slice
    
        # Load data
        image = (imread(f'{root_dir}{EVAL_DIR}{cell_id}/t{t_lab}.tif')[np.newaxis]).astype(np.float32)
        mask = (imread(f'{root_dir}{EVAL_DIR}{cell_id}_SEG/SEG/man_seg{t_lab}.tif')[np.newaxis] > .5) # volume data
        
        # Normalise Images
        T1 = transforms.Normalize(mean=(image.mean(),),
                                  std=(image.std(),))            
        image.astype(np.float32)            
        image, _ = T1(image, [])
        
        # Run model
        inp = torch.from_numpy(image[np.newaxis]).float().to(device)
        with torch.no_grad():  # Save memory by disabling gradients
            output = trainer.model(inp).cpu()
        probs = output.softmax(1).detach().numpy()
        
        prediction = np.array(np.argmax(probs, axis=1)) > 0.5
        prediction = custom_post_process(prediction)

        # Jaccard Scores
        union = np.maximum(prediction[:,zs], mask)
        intersect = np.zeros_like(mask); intersect[(prediction[:,zs]==1)&(mask==1)] = 1
        
        objective = np.sum(intersect)/np.sum(union)

        score_list.append(objective)

        try:
            results[cell_id][t_lab+'_'+zs_lab] = objective
        except:
            results[cell_id] = {} 
            results[cell_id][t_lab+'_'+zs_lab] = objective

        all_scores.append(objective)

    else:
        raise Exception('Label Files not Valid. \n Must be man_seg000.tif for volume labels, or man_seg000_000.tif for slice labels.')

with open(f'{root_dir}{EVAL_DIR}results_{MODEL_NAME}.pkl','wb') as f:
    pickle.dump(results, f)
                            
                        