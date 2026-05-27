"""
StarDist Training using Synthetic Data, with Evaluation Step against Real Data.

Use config files found in config/ to train a model. 

NOTE: custom_post_process can be changed between the datasets (hole filling, smoothing, etc.)
      This function is imported to test_models.py when running final segmentation.

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import re
import pickle
import argparse

## Root for Custom Elektronn3
if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())
    
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

import glob
from tqdm import tqdm
from tifffile import imread
import h5py
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

from utils import load_config, fill_holes_by_slice, jaccard_per_instance_2, custom_post_process
from utils import load_h5, load_h5_instance, z_transform, random_intensity_change

from csbdeep.utils.tf import limit_gpu_memory
# adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
limit_gpu_memory(0.8, total_memory=12000)

np.random.seed(42)
lbl_cmap = random_label_cmap()

parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--disable-cuda', action='store_true', 
                    help='If True, train on CPU')
parser.add_argument('--data-root', type=str, required=True,
                    help='Path to the base directory containing all datasets.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--train-dir', type=str, required=True,
                    help='Directory with Training Data in default structure.')
parser.add_argument('--max-samples', default=None,
                    help='Specify Number of training samples to include from training dataset')
parser.add_argument('--eval-dir', default=None,
                    help='Directory with Training Data in default structure.')
parser.add_argument('--model-dir', type=str, required=True,
                    help='Directory to Save Trained Model.')
parser.add_argument('--model-name', type=str, required=True,
                    help='Model Name.')
parser.add_argument('--config', type=str, required=True,
                    help='YAML Config. File')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()


if args.data_root not in sys.path:
    (sys.path).append(args.data_root)

assert args.train_dir[-1] == '/'
if args.eval_dir is not None:
    assert args.eval_dir[-1] == '/'
assert args.model_dir[-1] == '/'

label_dir = '' ## image labels 1 level above textured images
for s in (args.train_dir).split('/')[:-2]:
    label_dir += s + '/'
    
config = load_config(args.config)
if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
aniso_factor = global_params['ANISOTROPIC_FACTOR']
DATASET_MEAN = global_params['TRAIN_MEAN']
DATASET_STD = global_params['TRAIN_STD']
NORMALISE_P = global_params['TRAIN_P']

# Training
EPOCHS = config['training']['EPOCHS']
BATCH_SIZE = config['training']['BATCH_SIZE']
PATCH_SHAPE = config['training']['PATCH_SHAPE']
MAX_STEPS = config['training']['MAX_STEPS']
MAX_RUNTIME = 24 * 3600 # sequential training steps means this doesn't matter
LOSS_CODE = config['training']['LOSS_CODE']
WEIGHTS = config['training']['WEIGHTS']
WARP_PROB = config['training']['WARP_PROB']
DATASET_MEAN_ZERO = False

# Evaluation - if args.eval_dir set
STEP_SIZE = config['evaluation']['STEP_SIZE']
BURN_IN = config['evaluation']['BURN_IN']
EVAL_TYPE = config['evaluation']['EVAL_TYPE']
EVAL_TARGET = config['evaluation']['EVAL_TARGET']
EVAL_PATIENCE = config['evaluation']['EVAL_PATIENCE']
EVAL_SCALE = config['evaluation']['EVAL_SCALE']
EVAL_LABEL = config['evaluation']['EVAL_LABEL']

# Model
OUT_CHANNELS = config['model']['OUT_CHANNELS']
KERNEL_SIZE = config['model']['KERNEL_SIZE']
N_BLOCKS = config['model']['N_BLOCKS']
START_FILTS = config['model']['START_FILTS']
PLANAR_BLOCKS = config['model']['PLANAR_BLOCKS']
if config['model']['RESUME'] == '-':
    RESUME = None # placeholder for resuming training
else:
    RESUME = config['model']['RESUME']

# Filtering
Z_MED_FILTER = config['filtering']['Z_MED_FILTER']
CONV_Z = config['filtering']['CONV_Z']


'''  Datasets  '''

print(args.train_dir)
# paths to data
X = sorted(glob.glob(f'{args.train_dir}image_*.h5'))
Y = sorted(glob.glob(f'{label_dir}mask_*.h5'))

# crop list if specified, and less than the number of training samples
if args.max_samples is not None:
    if int(args.max_samples) < len(X):
        X = X[:int(args.max_samples)]
        Y = Y[:int(args.max_samples)]

# load data
X = list(map(load_h5,X))
Y = list(map(load_h5_instance,Y))
n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]

# normalise images to dataset mean and std, and fill holes of labels
axis_norm = (0,1,2)
X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

# Training and Validation Datasets
assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.2 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))



''' Model Config '''

anisotropy = (aniso_factor / EVAL_SCALE, 1.0, 1.0)

# 96 is a good default choice (see 1_data.ipynb)
n_rays = 96

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

conf = Config3D (
    train_checkpoint       = 'model_best.h5',
    train_checkpoint_last  = 'model_last.h5',
    train_checkpoint_epoch = 'model_now.h5',
    rays                   = rays,
    grid                   = grid,
    anisotropy             = anisotropy,
    use_gpu                = use_gpu,
    n_channel_in           = n_channel,
    unet_n_depth           = N_BLOCKS,
    unet_kernel_size       = (KERNEL_SIZE,KERNEL_SIZE,KERNEL_SIZE),
    unet_n_conv_per_depth  = 1,
    unet_n_filter_base     = START_FILTS, 
    # adjust for your data below (make patch size as large as possible)
    train_patch_size       = tuple(PATCH_SHAPE),
    train_batch_size       = BATCH_SIZE,
)
# print(conf)  

model = StarDist3D(conf, name=args.model_name, basedir=args.model_dir)

median_size = calculate_extents(Y, np.median)
fov = np.array(model._axes_tile_overlap('ZYX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")


''' Augmentation '''

def augmenter(x, y): # Add deformations to this?
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x = random_intensity_change(x)

    return x, y


''' Training '''

# convert the hyperparameters to match Elektronn3

# eg.
# EPOCHS = 100
# STEP_SIZE = 500
# MAX_STEPS = 100000
# BURN_IN = 20000

epochs = int(STEP_SIZE/EPOCHS)
steps_per_epoch = EPOCHS

print(epochs)
print(steps_per_epoch)

total_training_steps = 0

eval_x = []
eval_y = []
moving_average_best_score = 0
patience_counter = 0
moving_average_eval_score = 0

while patience_counter < EVAL_PATIENCE and total_training_steps < MAX_STEPS:

    # run/continue the training up to total training steps
    # load_weights automatically loads best model based on validation
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    if total_training_steps >= BURN_IN and args.eval_dir is not None: # burn-in period until you start to get reasonable outputs
        EVAL_PATH = args.data_root + args.eval_dir
   
        ### Evaluate model on real training data
        man_seg_files = glob.glob(f'{EVAL_PATH}*_{EVAL_LABEL}/SEG/man_seg*.tif')

        if len(man_seg_files) > 20: # select 20 files at random to evaluate model progress
            man_seg_files = np.random.choice(man_seg_files, size=20, replace=False)
        
        score_list = [];
        print(f'Evaluating Model on {EVAL_PATH}images ...')
        for i, man_seg_file in enumerate(man_seg_files):

            model.load_weights('model_last.h5')
            
            if i%1==0:
                print(f'Processing Image {i+1}/{len(man_seg_files)}...')

            ## if man_segXXX.tif - volume, if man_segXXX_XXX.tif - slice
            N_vals = len(re.findall(r'\d+', man_seg_file.split('/')[-1]))

            if N_vals == 1: # Volume label

                cell_id, t_lab = re.findall(r'\d+', man_seg_file)[-2:]
            
                # Load data
                image = imread(f'{EVAL_PATH}{cell_id}/t{t_lab}.tif').astype(np.float32)
                try:
                    mask = imread(f'{EVAL_PATH}{cell_id}_{EVAL_LABEL}/SEG/man_seg{t_lab}.tif')[np.newaxis] # volume data
                except:
                    mask = imread(f'{EVAL_PATH}{cell_id}_{EVAL_LABEL}/SEG/man_seg_{t_lab}.tif')[np.newaxis] # volume data
                    
                if EVAL_SCALE > 1:
                    image = image[:,::EVAL_SCALE,::EVAL_SCALE]
                    mask = mask[:,::EVAL_SCALE,::EVAL_SCALE]
                
                # Normalise Image
                image = normalize(image,1,99.8,axis=(0,1,2))
                
                # Run model
                prediction = model.predict_instances(image, n_tiles=model._guess_n_tiles(image), 
                                                     show_tile_progress=False)[0]
                
                # prediction = custom_post_process(prediction)
        
                # Jaccard Scores
                union = np.maximum(prediction, mask)
                intersect = np.zeros_like(mask); intersect[(prediction==1)&(mask==1)] = 1
                
                objective = np.sum(intersect)/np.sum(union)
        
                score_list.append(objective)

            elif N_vals == 2: # Slice label

                cell_id, t_lab, zs_lab = re.findall(r'\d+', man_seg_file)[-3:]
                zs = int(zs_lab[:2].lstrip('0') + zs_lab[-1]) # z slice
            
                # Load data
                image = imread(f'{EVAL_PATH}{cell_id}/t{t_lab}.tif').astype(np.float32)
                try:
                    mask = imread(f'{EVAL_PATH}{cell_id}_{EVAL_LABEL}/SEG/man_seg{t_lab}_{zs_lab}.tif') # volume data
                except:
                    mask = imread(f'{EVAL_PATH}{cell_id}_{EVAL_LABEL}/SEG/man_seg_{t_lab}_{zs_lab}.tif') # volume data

                if EVAL_SCALE > 1:
                    image = image[:,::EVAL_SCALE,::EVAL_SCALE]
                    mask = mask[::EVAL_SCALE,::EVAL_SCALE]
                
                # Normalise Image
                image = normalize(image,1,99.8,axis=(0,1,2))
                
                # Run model
                prediction = model.predict_instances(image, n_tiles=model._guess_n_tiles(image), 
                                                     show_tile_progress=False)[0]
        
                # # Jaccard Scores
                # union = np.maximum(prediction[:,zs], mask)
                # intersect = np.zeros_like(mask); intersect[(prediction[:,zs]==1)&(mask==1)] = 1
                
                # objective = np.sum(intersect)/np.sum(union)

                ## Jaccard per instance
                score_dict = jaccard_per_instance_2(prediction[zs].astype(int), mask.astype(int))[1]

                score_list += [score_dict[s] for s in score_dict if score_dict[s] > .5] # only add the score of the matches

            else:
                raise Exception('Label Files not Valid. \n Must be man_seg000.tif for volume labels, or man_seg000_000.tif for slice labels.')
            
        # JaccardScore
        if len(score_list) == 0: # no matches
            mean_objective = 0
        else:
            mean_objective = np.mean(score_list)

        # only update the training progress if the mean_objective is above some threshold (brute force)
        if mean_objective > .1:

            total_training_steps += STEP_SIZE # update training steps

            # update model evaluation and figure
            eval_x.append(total_training_steps)
            eval_y.append(mean_objective)
            eval_score = 1*mean_objective
    
            if len(eval_y) < EVAL_PATIENCE:
                moving_average_eval_score = np.sum(eval_y)/EVAL_PATIENCE
            else:
                moving_average_eval_score = np.sum(eval_y[-EVAL_PATIENCE:])/EVAL_PATIENCE
    
            print(f'\nEvaluation: {mean_objective}')
            print(f'Current Moving Average: {moving_average_eval_score}')
    
            if EVAL_TYPE == 'maximise':
                if moving_average_eval_score > moving_average_best_score:
                    moving_average_best_score = moving_average_eval_score
                    patience_counter = 0
                    print(f'New Moving Average Best: {moving_average_eval_score}')
                    print(f'Patience Counter Rest.')
                    ## save the best model
                    model.keras_model.save_weights(args.model_dir+args.model_name+'/eval_best.h5')
                
                if moving_average_best_score < EVAL_TARGET:
                    print(f'Patience Counter Rest.')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'Patience Counter: {patience_counter}\n')
                    
            elif EVAL_TYPE == 'minimise':
                if moving_average_eval_score < moving_average_best_score:
                    moving_average_best_score = moving_average_eval_score
                    patience_counter = 0
                    print(f'New Moving Average Best: {moving_average_eval_score}')
                    print(f'Patience Counter Rest.')
                
                if moving_average_best_score > EVAL_TARGET:
                    print(f'Patience Counter Rest.')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'Patience Counter: {patience_counter}\n')
                
            else:
                raise Exception('EVAL_TYPE shouldbe \'maximise\' or \'minimise\'.')
    
            # update outputs for plotting
            evaluation_outputs = {'train_steps': np.array(eval_x), 'model_eval': np.array(eval_y)}
            with open(f'{args.model_dir}{args.model_name}/evaluation_outputs.pkl', 'wb') as f:
                pickle.dump(evaluation_outputs, f)

        else:
            print('Retraining Failed Model!')

    else: # if burn-in not met (or eval not set to True)
        total_training_steps += STEP_SIZE # update training steps
































