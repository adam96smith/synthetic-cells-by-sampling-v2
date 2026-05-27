'''
Synthetic Image Generator

Fixed method for generating synthetic image samples using GeneratorSampler.py and GeneratorMask.py outputs.

'''

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
import os
import glob
import h5py
from tqdm import tqdm
import pickle
import sys
import random

# random_seed = 0
# np.random.seed(random_seed)

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())
    
from synthetic_generator import texture_mask
from my_elektronn3.custom.perlin_noise import *

from utils import load_config, max_down_sample, apply_z_varying_psf, dataset_properties

import argparse

parser = argparse.ArgumentParser(description='Generate Synthetic Images from Reference Labels.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--mask-dir', type=str, required=True,
                    help='Path to 3D Masks')
parser.add_argument('--sampler-dir', type=str, required=True,
                    help='Directory with Sampled Fluorescence')
parser.add_argument('--all-samplers',action='store_true',
                    help='Use all Samplers')
parser.add_argument('--one-label',action='store_true',
                    help='Set all mask labels to 1')
parser.add_argument('--resample',  action='store_true', 
                    help='Activate to resample original fluorescence to ground truth mask')
parser.add_argument('--sub-folder', type=str, default='custom_texture/', 
                    help='Target Directory for textured Images')
parser.add_argument('--config', type=str, required=True,
                    help='Config. File for Synthetic Images')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()

assert args.mask_dir[-1] == '/'
assert args.sampler_dir[-1] == '/'
assert args.sub_folder[-1] == '/'

''' Function '''

def balanced_random_sample(items, N):
    n = len(items)
    base_count = N // n
    extra = N % n

    # Start with base_count copies of each item
    samples = items * base_count

    # Randomly choose 'extra' items to get one additional sample
    extras = random.sample(items, extra)
    samples.extend(extras)

    # Shuffle the result
    random.shuffle(samples)
    return samples

def next_power_of_two(x):
    """
    Compute the next power of 2 greater than or equal to x. Use this to 
    make sure the perlin noise generator works for any input shape.
    
    Args:
        x (int): Input value (must be >= 1).
        
    Returns:
        int: The next power of 2 greater than or equal to x.
    """
    if x < 1:
        raise ValueError("Input must be greater than or equal to 1.")
    
    # If x is already a power of 2, return x
    if (x & (x - 1)) == 0:
        return x

    # Compute the next power of 2
    return 1 << (x - 1).bit_length()

def npt_seq(X):
    output = []
    for x in X:
        output.append(next_power_of_two(x))
    return output

def pN_norm(arr):
    '''
    Normalise Perlin Noise to [-1,1]
    '''

    outp = (arr-arr.min())/(arr.max()-arr.min())
    outp *= 2
    outp += -1

    return outp 


''' PARAMETERS '''

config = load_config(args.config)
if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']
ignore_sampler_list = global_params['IGNORE_SAMPLERS']

sf = config['SHAPE']['DOWNSAMPLE']

warp_dist_p = config['AUGMENTATION']['WARP_DIST_P']
warp_dist_A = config['AUGMENTATION']['WARP_DIST_A']
warp_norm_p = config['AUGMENTATION']['WARP_NORM_P']
warp_norm_A = config['AUGMENTATION']['WARP_NORM_A']

distmap_blur = config['SAMPLER']['DISTMAP_BLUR']
distmap_sig = config['SAMPLER']['DISTMAP_SIG']
partition_jitter = config['SAMPLER']['PARTITION_JITTER']
gaussian_blur = config['SAMPLER']['GAUSSIAN_BLUR']
gaussian_sig = config['SAMPLER']['GAUSSIAN_SIG']
normalise = config['SAMPLER']['NORMALISE']

if sf > 1: # scale sampling rates
    sampling = (sampling[0], sampling[1]*sf, sampling[2]*sf)

''' MAIN '''


# input directory
mask_files = sorted(glob.glob(args.mask_dir+'mask_*.h5'))
N = len(mask_files)

# output directory
os.makedirs(f'{args.mask_dir}{args.sub_folder}', exist_ok=True)

# sampler files (ignore poor samples)
if args.all_samplers:
    sampler_files = sorted(glob.glob('data_generator/sampled_data/'+args.sampler_dir+'*.pkl'))
else:
    print(f'Ignoring {len(ignore_sampler_list)} Selected Samplers')
    sampler_files = sorted([s for s in glob.glob('data_generator/sampled_data/'+args.sampler_dir+'*.pkl') if s.split('/')[-1] not in ignore_sampler_list])

if args.resample:
    sampler_inds = np.arange(N) # this is correct, index not number in file name
else:
    sampler_inds = balanced_random_sample(list(np.arange(len(sampler_files))), N)

metadata = {}
for i in tqdm(range(N), desc='Generating Image Textures'):

    sampler_idx = sampler_inds[i]
    mask_file = mask_files[i]

    # load data
    with h5py.File(mask_file,'r') as f:
        try:
            mask_labels = f['instance'][...]
        except:
            mask_labels = f['data'][...].astype(int)

    if args.one_label:
        mask_labels = (mask_labels>0).astype(int)

    # Calculate the new full distance map
    dist_map, indices = distance_transform_edt(mask_labels[0]==0, sampling=sampling, return_indices=True)
    for j in range(mask_labels.max()):
        dist_map += -distance_transform_edt(mask_labels[0]==j+1, sampling=sampling)
    partitions = mask_labels[0][tuple(indices)].astype(np.uint8)

    if np.random.rand() < warp_dist_p:

        grid_points = np.random.choice([2**i for i in range(1,10) if 2**i < np.min(mask_labels.shape[1:])/2])
        pN_1 = perlin_optim_3d(npt_seq(mask_labels.shape[1:]), [grid_points,grid_points,grid_points], octaves=1, sf=2, order=1)
        pN_1 = np.exp( warp_dist_A*pN_norm( pN_1[:mask_labels.shape[1],:mask_labels.shape[2],:mask_labels.shape[3]] ) )
        
        dist_map = np.minimum(dist_map * pN_1, 999+0*dist_map)
    
    # Load the sampler
    # print(sampler_files[sampler_idx])
    with open(sampler_files[sampler_idx],'rb') as f:
        sampler = pickle.load(f)
        
    image = texture_mask((mask_labels>0)[0], 
                         sampler, 
                         dist_map=dist_map, 
                         labelled_mask=mask_labels[0],
                         partitions=partitions,
                         sampling=sampling, 
                         distmap_blur=distmap_blur, 
                         distmap_sig=distmap_sig, 
                         gaussian_blur=gaussian_blur, 
                         gaussian_sig=gaussian_sig,
                         jitter_sigma=partition_jitter
                        )
    image = image.astype(np.float32)

    if np.random.rand() < warp_norm_p:
        
        grid_points = np.random.choice([2**i for i in range(1,10) if 2**i < np.min(mask_labels.shape[1:])/2])
        pN_2 = perlin_optim_3d(npt_seq(mask_labels.shape[1:]), [grid_points,grid_points,grid_points], octaves=1, sf=2, order=1)
        pN_2 = np.exp( warp_norm_A*pN_norm( pN_2[:mask_labels.shape[1],:mask_labels.shape[2],:mask_labels.shape[3]] ) )
        
        bg_f = np.mean(image[mask_labels[0]==0])
        img_std = np.std(image) * pN_2
        
        image = np.std(image) * (image-bg_f)/img_std + bg_f
            
    if normalise:
        image = (image-image.mean())/image.std() # normalise image

    with h5py.File(f'{args.mask_dir}{args.sub_folder}image_{str(i+1).zfill(5)}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=image[np.newaxis].astype(np.float32))

    
    metadata[str(i+1).zfill(5)] = {'sampler': sampler_files[sampler_idx].split('/')[-1]}

with open(f'{args.mask_dir}{args.sub_folder}metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)


## Calculate Dataset Properties
im_list = glob.glob(f'{args.mask_dir}{args.sub_folder}image_*.h5')

dataset_mean, dataset_std = dataset_properties(im_list)

with open(f'{args.mask_dir}{args.sub_folder}data_props.pkl', 'wb') as f:
    pickle.dump({'mean':dataset_mean, 'std':dataset_std}, f)































