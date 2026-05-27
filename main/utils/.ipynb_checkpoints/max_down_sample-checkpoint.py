import numpy as np

def max_down_sample(mask, aniso_factor):
    mask_new = np.reshape(mask, (aniso_factor, mask.shape[0]//aniso_factor, mask.shape[1], mask.shape[2]), order='F')
    return np.max(mask_new, axis=0)