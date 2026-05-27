'''
Image Label Generator

NOTE: This is to be customised between datasets. The goal is to match the size, numbers, 
      density and shape of the cells in the real images. Branches and Invaginations might
      be key features to add.

Output: .h5 file of binary image ('data') and labelled image ('instance') required for 
        GeneratorImage.py and GeneratorCurveSeg.py (if applicable).

'''

import numpy as np
import os
import tifffile
from tqdm import tqdm
import pickle
import sys

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

from scipy.ndimage import distance_transform_edt, label
from utils import load_config, nearest_power_of_two
from utils.ept_transform import ElasticPerlinTransform
from skimage.morphology import remove_small_objects

import argparse

parser = argparse.ArgumentParser(description='Generate Synthetic Cell Shapes.')
parser.add_argument('--save-path', type=str, default='synthetic_train/', 
                    help='Output Directory')
parser.add_argument('--N', type=int, default=100, 
                    help='Number of Total Generated Samples.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--config', type=str, required=True,
                    help='Config. File for Synthetic Images')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()

''' PARAMETERS '''

config = load_config(args.config)
if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']
aniso_factor = nearest_power_of_two(sampling[0]/sampling[1])

zres, xres, yres  = config['SHAPE']['IMAGE_SIZE']
sf  = config['SHAPE']['DOWNSAMPLE']
n0, n1 = config['SHAPE']['CELL_N']
r0, r1 = config['SHAPE']['CELL_R']
z_scale = config['SHAPE']['Z_SCALE']
r_sep = config['SHAPE']['CELL_SEPARATION']
P = config['SHAPE']['CELL_COUPLING_CHANCE']
ept1 = config['SHAPE']['DEFORM_1']
ept2 = config['SHAPE']['DEFORM_2']

if sf > 1:
    sampling = (sampling[0], sf*sampling[1], sf*sampling[2])

## Define Box in Image to Add Cells such that none cross the boarder (essential for sampling method)
z0 = int(z_scale*r1/sampling[0]); z1 = zres - z0
x0 = int(r1/sampling[1]); x1 = xres - x0
y0 = int(r1/sampling[2]); y1 = yres - y0

print(f'Z: {z0} - {z1}')
print(f'X: {x0} - {x1}')
print(f'Y: {y0} - {y1}')

''' MAIN '''

assert args.save_path[-1] == '/'
os.makedirs(args.save_path, exist_ok=True)

if ept1 is not False:
    EPT1 = ElasticPerlinTransform(prob=1,
                                  grid_points=ept1[0],
                                  p=ept1[1],  order=1) # small
if ept2 is not False:
    EPT2 = ElasticPerlinTransform(prob=1,
                                  grid_points=ept2[0],
                                  p=ept2[1],  order=1) # large

# initialise image and sample space
Z, X, Y = np.meshgrid(np.arange(0,zres), np.arange(0,xres), np.arange(0,yres), indexing='ij')
sZ, sX, sY = np.meshgrid(np.arange(z0,z1), np.arange(x0,x1), np.arange(y0,y1), indexing='ij')

metadata = {}
for i in tqdm(range(args.N), desc='Generating Image Masks'):

    N = np.random.choice(np.arange(n0,n1+1))

    # radius of first cell
    r_c = np.random.uniform(r0,r1)

    # initiate sample space
    sample_space = np.ones(sZ.shape, bool)
    
    centroids = []
    radii = []
    counter = 0
    while counter < N and sample_space.sum() > 0:

        radii.append(r_c) # add cell to metadata
    
        # all possible coordinates for new cell
        z_space = sZ[sample_space];  x_space = sX[sample_space]; y_space = sY[sample_space]
        coords = np.vstack((z_space,x_space,y_space))
        
        # randomly select a coordinate
        j = np.random.choice(coords.shape[1])         
        z_c = z_space[j];  x_c = x_space[j]; y_c = y_space[j]
    
        centroids.append([z_c, x_c, y_c])
    
        if counter == 0:
            dist = np.sqrt(((Z-z_c)*sampling[0]/z_scale)**2+((X-x_c)*sampling[1])**2+((Y-y_c)*sampling[1])**2) - r_c
            dist_s = np.sqrt(((sZ-z_c)*sampling[0]/z_scale)**2+((sX-x_c)*sampling[1])**2+((sY-y_c)*sampling[1])**2) - r_c
        else:
            dist = np.minimum(dist, np.sqrt(((Z-z_c)*sampling[0]/z_scale)**2+((X-x_c)*sampling[1])**2+((Y-y_c)*sampling[1])**2) - r_c)
            dist_s = np.minimum(dist_s, np.sqrt(((sZ-z_c)*sampling[0]/z_scale)**2+((sX-x_c)*sampling[1])**2+((sY-y_c)*sampling[1])**2) - r_c)
    
        counter += 1
    
        # next cell to add        
        r_c = np.random.uniform(r0,r1)

        if np.random.rand() < P: # couple cells
            sample_space = np.abs(dist_s - .8*r_c) <= sampling[0] 
            # sample_space = np.abs(dist_s - r_sep) <= sampling[0] 
        else:
            sample_space = dist_s > r_sep*r_c

    # Generate Mask labels - split membrane and reassign nearest interior label
    outline = (-3<dist)&(dist<=0) # not sure if 3 is a robust choice
    interior = label(dist<=-3)[0]
    
    dist_map, indices = distance_transform_edt(interior==0, sampling=sampling, return_indices=True)
    
    mask_labels = (dist<=0) * interior[tuple(indices)]

    # deform labelled image
    if ept1 is not False:
        mask_labels = EPT1(mask_labels[np.newaxis], [])[0][0] # small
    if ept2 is not False:
        mask_labels = EPT2(mask_labels[np.newaxis], [])[0][0] # large

    ## filter small objects
    output = np.zeros_like(mask_labels)

    current_id = 1
    for cls in np.unique(mask_labels)[1:]:
        cc, n = label(remove_small_objects(mask_labels == cls, 10))
        for j in range(1, n + 1):
            output[cc == j] = current_id
            current_id += 1
        

    
    tifffile.imwrite(f'{args.save_path}mask_{str(i+1).zfill(5)}.tif', output.astype(np.uint16))
    
    metadata[str(i+1).zfill(5)] = {'N': N,
                                   'radii': radii, 
                                   }

with open(f'{args.save_path}metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

