"""
Model Training using Synthetic Data, with Evaluation Step against Real Data.

Use config files found in config/ to train a model. 

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
from tifffile import imread

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch import nn
from torch import optim
import numpy as np
import math
import pickle

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
from my_elektronn3.modules import CombinedLoss, CurvatureWeightedLoss, DiceLoss
from my_elektronn3.custom.image_filtering import MedianFiltering, GaussianFiltering, ConvZFiltering

# from boundary_loss import BoundaryLoss # from https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py
from my_elektronn3.models.unet import UNet

from utils import load_config, nearest_power_of_two, dataset_properties, evaluate_during_training


parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--disable-cuda', action='store_true', 
                    help='If True, train on CPU')
parser.add_argument('--data-root', type=str, required=True,
                    help='Path to the base directory containing all datasets.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--train-dir', type=str, required=True,
                    help='Directory with Training Data in default structure.')
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
sampling = global_params['SAMPLING']
normalised_sampling = global_params['SAMPLER']['NORMALISE']
aniso_factor = nearest_power_of_two(sampling[0]/sampling[1])

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
EVAL_DATA = config['evaluation']['EVAL_DATA']
EVAL_TYPE = config['evaluation']['EVAL_TYPE']
EVAL_TARGET = config['evaluation']['EVAL_TARGET']
EVAL_PATIENCE = config['evaluation']['EVAL_PATIENCE']
EVAL_SCALE = config['evaluation']['EVAL_SCALE']
EVAL_ID = config['evaluation']['EVAL_ID']

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
if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')        
# logger.info(f'Running on device: {device}')

# inputs
input_h5data = []
for f in sorted(glob.glob(f'{args.train_dir}image_*.h5')):
    input_h5data.append((f, 'data'))    
    
# ground truth
target_h5data = []
for f in sorted(glob.glob(f'{label_dir}mask_*.h5')):
    target_h5data.append((f, 'data'))  


# default for new model
optimizer_state_dict = None
lr_sched_state_dict = None

if normalised_sampling == True:
    dataset_mean, dataset_std = 0.0, 1.0
else:
    try:
        with open(f'{args.train_dir}data_props.pkl', 'rb') as f:
            dataset_props = pickle.load(f)
            dataset_mean = dataset_props['mean']
            dataset_std = dataset_props['std']
    except:
        dataset_mean, dataset_std = dataset_properties(glob.glob(f'{args.train_dir}image_*.h5'))
    
aniso_train = aniso_factor/EVAL_SCALE

common_data_kwargs = {'aniso_factor': aniso_train,
                      'patch_shape': PATCH_SHAPE,}


common_transforms = transforms.Compose([transforms.SqueezeTarget(dim=0),
                                        transforms.Normalize(mean=dataset_mean, std=dataset_std),
                                        transforms.RandomScaleShift(scale=.25, shift=.1),
                                        transforms.ElasticPerlinTransform(prob=WARP_PROB,grid_points=[1,4,4],p=.075),
                                        transforms.RandomGaussianBlur(prob=1, distsigma=2, aniso_factor = aniso_train/4),
                                        transforms.AdditiveGaussianNoise(sigma=1, prob=WARP_PROB),
                                        ])

image_filters = list()
if Z_MED_FILTER != [1,1,1]:
    image_filters.append(MedianFiltering(size=tuple(Z_MED_FILTER)))
if CONV_Z != [0,1]:
    image_filters.append(ConvZFiltering(kern=CONV_Z))
    

# Patch Creators    
valid_indices = random.sample(list(np.arange(len(input_h5data))), int(.2*len(input_h5data)))    

train_sources = [[target_h5data[i]] for i in range(len(input_h5data)) if i not in valid_indices]
valid_sources = [[target_h5data[i]] for i in range(len(input_h5data)) if i in valid_indices]

train_dataset = PatchCreator(input_sources=[input_h5data[i] for i in range(len(input_h5data)) if i not in valid_indices],
                             target_sources=train_sources,
                             train=True,
                             epoch_size=EPOCHS,
                             device=device,
                             warp_kwargs={'sample_aniso': aniso_train != 1,},
                             transform=common_transforms,
                             filters=image_filters,
                             in_memory=True,
                             **common_data_kwargs
                             )

valid_dataset = PatchCreator(input_sources=[input_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
                             target_sources=valid_sources,
                             train=False,
                             epoch_size=EPOCHS,
                             device=device,
                             warp_kwargs={'sample_aniso': aniso_train != 1},
                             transform=common_transforms,
                             filters=image_filters,
                             in_memory=True,
                             **common_data_kwargs
                             )
print('All Data Loaded.') 

''' Create UNet '''    
if RESUME is not None:
    print(f'Resuming {RESUME}...')
    model = torch.load(RESUME, map_location=device, weights_only=False)
else:
    model = UNet(
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        n_blocks=N_BLOCKS,
        start_filts=START_FILTS,
        planar_blocks=PLANAR_BLOCKS,
        activation='relu',
        normalization='batch',
        conv_mode='same',
        dim=3,
    ).to(device)

# model save directory
os.makedirs(args.model_dir, exist_ok=True)

# Set Image Filters
train_dataset.filters = image_filters 
valid_dataset.filters = image_filters

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=.05,  # Learning rate is set by the lr_sched below
    weight_decay=0.5e-4,
)
# optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging

#not sure on the steps here?
lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                             base_lr=1e-6, # 0.0001
                                             max_lr=1e-3, # 0.01
                                             step_size_up=500,
                                             step_size_down=1500,
                                             cycle_momentum=True if 'momentum' in optimizer.defaults else False
                                             )

# Validation metrics
valid_metrics = {}
for evaluator in [metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.DSC, metrics.IoU]:
    valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()  # Mean metrics
    for c in range(2):
        valid_metrics[f'val_{evaluator.name}_c{c}'] = evaluator(c)

## Loss Function
class_weights = torch.tensor([WEIGHTS]).to(device)

losses = []
loss_weights = []

pattern = r'(\d+)([A-Z]+)'
matches = re.findall(pattern, LOSS_CODE)

d = {key: int(value)/100 for value, key in matches}

if sum(d.values()) != 1:
    raise ValueError("The numbers must add up to 100.")

for loss_type in d:

    if loss_type == 'CE':        
        print('Adding Cross-Entropy Loss')
        losses.append(nn.CrossEntropyLoss(weight=class_weights))
        loss_weights.append(d[loss_type])

    elif loss_type == 'DL' or loss_type == 'GDL':
        print('Adding Dice Loss')
        losses.append(DiceLoss(apply_softmax=True, weight=class_weights))
        loss_weights.append(d[loss_type])

    elif loss_type == 'BL':
        print('Adding Boundary Loss')
        losses.append(BoundaryLoss(apply_softmax=True))
        loss_weights.append(d[loss_type])

    elif loss_type == 'CWL':
        print('Adding Curvature Weighted Cross-Entropy Loss')
        losses.append(CurvatureWeightedLoss(weights=WEIGHTS))
        loss_weights.append(d[loss_type])

criterion = CombinedLoss(criteria=losses, weight=loss_weights, device=device)


''' Build Trainer '''
trainer = Trainer(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  device=device,
                  train_dataset=train_dataset,
                  valid_dataset=valid_dataset,
                  batch_size=2,
                  save_root=args.model_dir,
                  save_full=False, # only save model.pt
                  exp_name = args.model_name,
                  schedulers={'lr': lr_sched},
                  valid_metrics=valid_metrics,
                  enable_tensorboard=False,
                  out_channels=2,
)

total_training_steps = 0

eval_x = []
eval_y = []
tr_loss = []
val_loss = []

absolute_best_score = 0
moving_average_best_score = 0
patience_counter = 0
moving_average_eval_score = 0

start_training = time.time()
while patience_counter < EVAL_PATIENCE and total_training_steps < MAX_STEPS:

    total_training_steps += STEP_SIZE # update training steps
    
    step_time = time.time()
    trainer.terminate = False # enable continue training 
    trainer.run(max_steps=total_training_steps, max_runtime=MAX_RUNTIME, plot_history=False) # run/continue the training up to total training steps
    
    training_time_minutes = (time.time() - start_training) / 60
    step_time_minutes = (time.time() - step_time) / 60
    print(f'\n\nTotal Run Time (minutes): {training_time_minutes:.2f}')
    print(f'\n\nTotal Step Time (minutes): {step_time_minutes:.2f}')
    print(f'\n\nTotal Steps : {total_training_steps}')
    
    if total_training_steps >= BURN_IN and args.eval_dir is not None: # burn-in period until you start to get reasonable outputs

        trainer.model.eval()

        if EVAL_DATA == 'CTC':
            mask_list = sorted(glob.glob(f'{args.data_root}{args.eval_dir}/*_GT/SEG/man_seg*.tif'))

            image_list = []
            for s in mask_list:
                N_vals = len(re.findall(r'\d+', s.split('/')[-1]))
                if N_vals == 1: # volume data
                    cell_id, t_lab = re.findall(r'\d+', s)[-2:]
                    image_list.append(f'{args.data_root}{args.eval_dir}{cell_id}/t{t_lab}.tif')
                else: # slice labels
                    cell_id, t_lab, _ = re.findall(r'\d+', s)[-3:]
                    image_list.append(f'{args.data_root}{args.eval_dir}{cell_id}/t{t_lab}.tif')

        elif EVAL_DATA == 'synth':
            mask_dir = ''
            for s in (args.eval_dir).split('/')[:-2]:
                mask_dir += s + '/'            
            mask_list = sorted(glob.glob(f'{args.data_root}{mask_dir}mask*.h5'))
            image_list = sorted(glob.glob(f'{args.data_root}{args.eval_dir}image*.h5'))
            
        else:
            raise Exception('Only CTC and synth formats supported for evaluation.')

        if len(image_list) > 50:
            i_list = np.random.choice(np.arange(len(image_list)), size=50, replace=False)
            mask_list = [f for i, f in enumerate(mask_list) if i in i_list]
            image_list = [f for i, f in enumerate(image_list) if i in i_list]

        try: # calculate the eval image properties if not already executed
            eval_data_mean 
        except:
            eval_data_mean, eval_data_std = dataset_properties(image_list)

        mean_objective = evaluate_during_training(trainer.model, 
                                                  image_list,
                                                  mask_list,
                                                  device, 
                                                  dataset_mean=eval_data_mean,
                                                  dataset_std=eval_data_std,
                                                  downsample_factor=EVAL_SCALE,
                                                  dataset_id=EVAL_ID)


        # update model evaluation and figure
        eval_x.append(total_training_steps)
        eval_y.append(mean_objective)
        tr_loss.append(trainer.last_tr_loss)
        val_loss.append(trainer.last_val_loss)
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
                # save best model
                trainer._save_model(suffix='_best', verbose=False)
            
            if moving_average_best_score < EVAL_TARGET:
                print(f'Patience Counter Rest.')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'Patience Counter: {patience_counter}\n')

            if mean_objective > absolute_best_score:
                absolute_best_score = 1*mean_objective
                print(f'New Absolute Best: {absolute_best_score}')
                trainer._save_model(suffix='_peak', verbose=False)
                
        elif EVAL_TYPE == 'minimise':
            if moving_average_eval_score < moving_average_best_score:
                moving_average_best_score = moving_average_eval_score
                patience_counter = 0
                print(f'New Moving Average Best: {moving_average_eval_score}')
                print(f'Patience Counter Rest.')
                # save best model
                trainer._save_model(suffix='_best', verbose=False)
            
            if moving_average_best_score > EVAL_TARGET:
                print(f'Patience Counter Rest.')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'Patience Counter: {patience_counter}\n')

            if mean_objective < absolute_best_score:
                absolute_best_score = 1*mean_objective
                print(f'New Absolute Best: {absolute_best_score}')
                trainer._save_model(suffix='_peak', verbose=False)
            
        else:
            raise Exception('EVAL_TYPE shouldbe \'maximise\' or \'minimise\'.')

        # update outputs for plotting
        evaluation_outputs = {'train_steps': np.array(eval_x), 'model_eval': np.array(eval_y), 'tr_loss': np.array(tr_loss), 'val_loss': np.array(val_loss)}
        with open(f'{args.model_dir}{args.model_name}/evaluation_outputs.pkl', 'wb') as f:
            pickle.dump(evaluation_outputs, f)

        trainer.model.train()



