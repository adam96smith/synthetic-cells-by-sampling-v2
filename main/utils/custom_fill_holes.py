'''
Custom Fill Holes

Using scipy.ndimage.binary_fill_holes, we fill binary output of cells overlapping cell boundary.

Method:
For 3D input, we mask image boundaries (1-pixel width) at 8 corners to fill holes. 
The final is a max-projection of 8 images (+ erosion + dilation to remove boundary).

'''

import numpy as np
from scipy import ndimage

def custom_fill_holes(mask, planar=False):

    # 8 masked inputs
    zres, xres, yres = mask.shape
    mask_list = []

    if planar: # Apply to x,y-axes only
        for i in range(2):
            for j in range(2):
                    tmp = 1*mask
    
                    # Set sides=1
                    tmp[:,i*(xres-1),:] = 1
                    tmp[:,:,j*(yres-1)] = 1
                    
                    mask_list.append(tmp.astype(bool))

    else: # Apply to all axis
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    tmp = 1*mask
    
                    # Set sides=1
                    tmp[i*(zres-1),:,:] = 1
                    tmp[:,j*(xres-1),:] = 1
                    tmp[:,:,k*(yres-1)] = 1
                    
                    mask_list.append(tmp.astype(bool))

    filled_mask = np.array([ndimage.binary_fill_holes(m) for m in mask_list]).max(axis=0)

    if planar:
        kern = np.array([[0,1,0],[1,1,1],[0,1,0]], bool)
        filled_mask = ndimage.binary_erosion(filled_mask, kern[np.newaxis])
        filled_mask = ndimage.binary_dilation(filled_mask, kern[np.newaxis])
    else:
        filled_mask = ndimage.binary_erosion(filled_mask)
        filled_mask = ndimage.binary_dilation(filled_mask)

    return filled_mask