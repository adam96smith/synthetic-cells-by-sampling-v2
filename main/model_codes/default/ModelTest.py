"""
Test trained Model on held-out synthetic data



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
from utils import load_config, upscale_by_slice, custom_post_process

parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--disable-cuda', action='store_true', 
                    help='If True, train on CPU')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--test-dir', type=str, required=True,
                    help='Directory with Test Data.')
parser.add_argument('--texture', type=str, required=True,
                    help='Texture within Test Directory.')
parser.add_argument('--model-dir', type=str, required=True,
                    help='Directory to Save Trained Model.')
parser.add_argument('--model-name', type=str, required=True,
                    help='Model Name.')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()


if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
aniso_factor = global_params['ANISOTROPIC_FACTOR']
DATA_MEAN = global_params['TRAIN_MEAN']
DATA_STD = global_params['TRAIN_STD']


if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')    
    
## Load Model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(f'{args.model_dir}{args.model_name}/model.pt', map_location=device)
model.eval()

### Evaluate model on real training data
N = len(glob.glob(f'{args.test_dir}{args.texture}*.h5'))
            
print(f'Evaluating Model on {args.test_dir}images...')

results = {}

for i in tqdm(range(N), desc='Evaluating Model'):

    image_file = f'{args.test_dir}{args.texture}image_{str(i+1).zfill(5)}.h5'
    mask_file = f'{args.test_dir}mask_{str(i+1).zfill(5)}.h5'
    
    # Load data
    with h5py.File(image_file,'r') as hf:
        dataset = hf['data']
        image = dataset[...]

    with h5py.File(mask_file,'r') as hf:
        dataset = hf['data']
        mask = dataset[...]
        

    T1 = transforms.Normalize(mean=(DATA_MEAN,),
                              std=(DATA_STD,))
    
    # T1 = transforms.Normalize(mean=(image.mean(),),
    #                           std=(image.std(),))
       
    image, _ = T1(image, [])
    
    # Run model
    inp = torch.from_numpy(image[np.newaxis]).float().to(device)
    with torch.no_grad():  # Save memory by disabling gradients
        output = model(inp).cpu()
    probs = output.softmax(1).detach().numpy()
    
    prediction = np.array(np.argmax(probs, axis=1)) > 0.5
    # prediction = custom_post_process(prediction)

    union = np.sum(np.maximum(prediction,mask))
    intersect = np.sum(prediction*mask)

    results[str(i+1).zfill(5)] = intersect/union

## Save Results
os.makedirs('test_results/', exist_ok=True)
os.makedirs(f'test_results/{args.dataset_id}/', exist_ok=True)

with open(f'test_results/{args.dataset_id}/{args.model_name}_results.pkl', 'wb') as f:
    pickle.dump(results, f)
    
        
    

                        