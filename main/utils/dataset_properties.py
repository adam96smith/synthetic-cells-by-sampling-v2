'''
Function for calculating the mean and standard deviation of a datasets.

Supported formats: .h5, .tiff
'''

import numpy as np
from tqdm import tqdm
from tifffile import imread
import h5py

# def dataset_properties(im_list):

#     counts = 0
#     sums = 0
#     sq_sums = 0
#     for i in tqdm(range(len(im_list)), desc='Calculating Dataset Props.'):
        
#         im_file = im_list[i]

#         if im_file.split('.')[-1] == 'tif':

#             image = imread(im_file)

#         elif im_file.split('.')[-1] == 'h5':

#             with h5py.File(im_file, 'r') as f:
#                 dataset = f['data']
#                 image = dataset[...]

#         counts += np.prod(image.shape)
#         sums += np.sum(image)
#         sq_sums += np.sum(image**2)

#     mu = sums/counts
#     std = np.sqrt((sq_sums/counts) - mu**2)

#     return mu, std

def dataset_properties(im_list):
    count = 0
    mean = 0.0
    M2 = 0.0  # sum of squares of differences from the current mean

    for im_file in tqdm(im_list, desc='Calculating Dataset Props.'):
        # load
        if im_file.endswith('.tif'):
            image = imread(im_file)
        elif im_file.endswith('.h5'):
            with h5py.File(im_file, 'r') as f:
                image = f['data'][...]

        data = image.astype(np.float64).ravel()
        n = data.size

        # online (Welford) update
        count_new = count + n
        delta = data.mean() - mean
        mean += delta * n / count_new
        M2 += data.var(ddof=0) * n + (delta**2) * count * n / count_new
        count = count_new

    std = np.sqrt(M2 / count)
    return mean, std