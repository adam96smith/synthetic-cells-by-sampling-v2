'''
Format real data for training input. This enables resampling flourescence on grount truth the insp[ect differentces between real and synthetic images
'''

import numpy as np
import glob
import os
import sys
import re
from tifffile import imread
import h5py
import argparse
from tqdm import tqdm
import pickle

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

from utils import load_config


parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--data-root', type=str, required=True,
                    help='Path to the base directory containing all datasets.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Directory with Training Data in default structure.')
parser.add_argument('--label', type=str, default='GT',
                    help='GT or ST (if available)')
parser.add_argument('--suffix', type=str, default='real')
parser.add_argument('--config', type=str, required=True,
                    help='Config. for Synth Data')
args = parser.parse_args()

config = load_config(args.config)

downsample = config['SHAPE']['DOWNSAMPLE']

# list of labelled images
man_seg_files = sorted(glob.glob(f'{args.data_root}{args.data_dir}*_{args.label}/SEG/man_seg*.tif'))

os.makedirs(f'synthetic_data/{args.dataset_id}_{args.suffix}/', exist_ok=True)
os.makedirs(f'synthetic_data/{args.dataset_id}_{args.suffix}/real/', exist_ok=True)

metadata = {}

for i in tqdm(range(len(man_seg_files)), desc='Labelled file:'):

    file = man_seg_files[i]

    assert len(re.findall(r'\d+', file.split('/')[-1])) == 1 # implies volume label

    cell_number, timepoint = re.findall(r'\d+', file)[-2:]

    image = imread(f'{args.data_root}{args.data_dir}/{cell_number}/t{timepoint}.tif') # int16
    mask = imread(file) # int16 labelled image

    # save in .h5 format for model training / image resampling

    with h5py.File(f'synthetic_data/{args.dataset_id}_{args.suffix}/real/image_{str(i+1).zfill(5)}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=image[np.newaxis, :, ::downsample, ::downsample].astype(np.float32))

    with h5py.File(f'synthetic_data/{args.dataset_id}_{args.suffix}/mask_{str(i+1).zfill(5)}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=(mask[np.newaxis, :, ::downsample, ::downsample]>0.5).astype(bool))
        dataset = hf.create_dataset('instance', data=mask[np.newaxis, :, ::downsample, ::downsample].astype(np.uint8))

    metadata[str(i+1).zfill(5)] = {'cell':cell_number, 't':timepoint}


with open(f'synthetic_data/{args.dataset_id}_{args.suffix}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)