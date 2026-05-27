'''
This script takes all model checkpoints and evaluates accuracy on 
annotated data for different confidence thresholds. The best value 
is saved for the test set.
'''


import numpy as np
import h5py
from tifffile import imread
import pickle
import torch
import glob
import os
import sys
import warnings
from tqdm import tqdm
import re
from scipy.ndimage import distance_transform_edt

## Root for Custom Elektronn3
if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

from utils import load_config, custom_post_process, nearest_power_of_two, jaccard_per_instance, upscale_by_slice

import argparse

parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--disable-cuda', action='store_true', 
                    help='If True, train on CPU')
parser.add_argument('--data-root', type=str, required=True,
                    help='Path to the base directory containing all datasets.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157, or custom_texture)')
parser.add_argument('--test-dir', type=str, required=True,
                    help='Directory with Training Data in default structure.')
parser.add_argument('--model', type=str, required=True,
                    help='Model Name.')
parser.add_argument('--config', type=str, required=True,
                    help='YAML Config. File')
parser.add_argument('--global-config', default=None,
                    help='YAML Global Config. File')
args = parser.parse_args()


config = load_config(args.config)
if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']
TEST_SCALE = config['evaluation']['EVAL_SCALE']

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

tv_list = np.arange(.1, .95+1e-5, .05)


if len(glob.glob(f'{args.data_root}{args.test_dir}*_GT/SEG/man_seg*.tif')) > 0:
    mask_file_list = sorted(glob.glob(f'{args.data_root}{args.test_dir}*_GT/SEG/man_seg*.tif'))
    data_type = 'CTC'
    
elif len(glob.glob(f'{args.data_root}{args.test_dir}image_*.h5')) > 0:
    mask_file_list = sorted(glob.glob(f'{args.data_root}mask_*.h5'))
    data_type = 'SYNTH'
    
else:
    raise Exception(f'Test Dataset Incorrectly Set - {args.data_root}{args.test_dir}* not found.')
    

print('Evaluating:')
print(f'- {args.model}')
print(f'On {len(mask_file_list)} annotated samples.')
print(f'{data_type}: {mask_file_list[0]} ... ')

## Load Model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(args.model, map_location=device, weights_only=False)
model.eval()
    
results = {'x': tv_list}

# Gather all model output probabilities
for i in tqdm(range(len(mask_file_list)), desc='Evaluating'):

    mask_file = mask_file_list[i]

    if data_type == 'CTC':
        mask = imread(mask_file)
        if len(re.findall(r'\d+', mask_file.split('/')[-1])) == 2:
            cell, tp, z_slice = re.findall(r'\d+', mask_file)[-3:]
            slice_label = True

        else:
            cell, tp = re.findall(r'\d+', mask_file)[-2:]
            slice_label = False

        image = imread(f'{args.data_root}{args.test_dir}{cell}/t{tp}.tif')
        if TEST_SCALE > 1:
            image = image[:,::TEST_SCALE,::TEST_SCALE]

    elif data_type == 'SYNTH':
        # Load Mask
        with h5py.File(mask_file, 'r') as f:
            mask = f['instance'][...][0]
        cell = re.findall(r'\d+', mask_file)[-2]
        
        # Load image
        with h5py.File(f'{args.data_root}{args.test_dir}image_{cell}.h5', 'r') as f:
            image = f['data'][...][0]
            
        # Synthetic Data Always 3D
        slice_label = False
    
    else:
        raise Exception('data_type error.')

    # Run model
    img = image[np.newaxis].astype(np.float32)
    
    img = (img-img.mean())/img.std()
    
    inp = torch.from_numpy(img[np.newaxis]).float().to(device)
    with torch.no_grad():  # Save memory by disabling gradients
        output = model(inp).cpu()
        
    probs = output.softmax(1).detach().numpy()

    if TEST_SCALE > 1: # apply scaling to probability map if necessary
        prediction = upscale_by_slice(probs[0,1], 
                                      sf=TEST_SCALE, 
                                      xy_target=mask.shape[-2:], 
                                      order=3,
                                     )
    else:
        prediction = probs[0,1]

    if slice_label:
        zs = int(re.findall(r'\d+', mask_file)[-1])
        predictions = [custom_post_process((prediction > tv)[np.newaxis], mode=args.dataset_id)[0,zs] for tv in tv_list]
    else:
        predictions = [custom_post_process((prediction > tv)[np.newaxis], mode=args.dataset_id)[0] for tv in tv_list]

    for pred in predictions:

        if slice_label:
            pred = pred.astype(int)
            
            if len(np.unique(mask)) > 2: # BG + 1 label > 0
                # use voronoi of GT to separate semantic segmentation
                _, indices = distance_transform_edt(mask==0, 
                                                    sampling=sampling[1:], 
                                                    return_indices=True)
                voronoi = 1*mask[tuple(indices)].astype(int)
                pred *= voronoi
                
            jaccard_data, _ = jaccard_per_instance(pred.astype(int), mask.astype(int))
        else:
            jaccard_data, _ = jaccard_per_instance(pred.astype(int), mask.astype(int))

        for s in jaccard_data:

            if data_type == 'CTC':
                if slice_label:
                    try:
                        results[f'{cell}_{tp}_{z_slice}_{s}'].append(jaccard_data[s])
                    except:
                        results[f'{cell}_{tp}_{z_slice}_{s}'] = [jaccard_data[s]]
                else:
                    try:
                        results[f'{cell}_{tp}_{s}'].append(jaccard_data[s])
                    except:
                        results[f'{cell}_{tp}_{s}'] = [jaccard_data[s]]

            elif data_type == 'SYNTH':
                try:
                    results[f'{cell}_{s}'].append(jaccard_data[s])
                except:
                    results[f'{cell}_{s}'] = [jaccard_data[s]]

            else:
                raise Exception('data_type error.')

# Find peak tv value
results_array = np.array([results[s] for s in results if s!='x'])
mean_results = np.mean(results_array, axis=0)

peak_tv = tv_list[np.argmax(mean_results)]

results['peak_tv'] = peak_tv

print(f'* Threshold {peak_tv:.3f} - Jaccard {np.max(mean_results):.3f} *')

# Save Data
model_dir = ''
for s in args.model.split('/')[:-1]:
    model_dir += s+'/'

with open(f'{model_dir}model_optim.pkl', 'wb') as f:
    pickle.dump(results, f)
    














