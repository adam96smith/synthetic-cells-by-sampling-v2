'''
Split by Curvature Method for splitting Cell Doublets

--- Method ---
1. Using a spherical kernel, calculate the curvature on the surface of a binary segmentation.
2. Threshold the negative curvature to highlight 'rings of contact' between two spherical objects
3. Remove the convex hull of rings, label the residual shapes and assign labels of contact areas.

--- Assumptions ---
1. Target objects are shperical-ish, or convex shapes
2. The contact region is reasonably flat.

--- Input ---
instances: Binary or Integer array of segmentation

--- Parameters ---
r: Radius of Spherical Kernel
threshold: Cut-off value for surface curvature for rings of contact
anisotropy: of input
iterations: number of dilation operations of ring (improves robustness)

'''

import numpy as np
from utils.curvature import curvature_approximation_3d
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

def quick_convex_hull(binary):

    labels = ndimage.label(binary)[0]
    # labels = binary.astype(int)

    output = np.zeros(binary.shape, bool)
    for R in regionprops(labels):
        z0, x0, y0, z1, x1, y1 = R.bbox
        if np.min([z1-z0, x1-x0, y1-y0]) > 1:
            output[z0:z1,x0:x1,y0:y1] += R.image_convex

    return output

def split_by_curvature(instances, r, threshold, aniso_factor=1, iterations=0):

    instances = instances.astpye(int)
    
    # for n in np.unique(instances):
    #     if n > 0:
    
    # calculate curvature on surface 
    curvature = curvature_approximation_3d(instances>0, r=r, aniso_factor=aniso_factor)

    if iterations > 0:
        split = binary_dilation(curvature < tv, iterations=iterations)
    else:
        split = binary_dilation(curvature < tv) # not recommended

    # convex hull 
    split = quick_convex_hull(split)

    # Remove the rings, label the residual segmentation
    tmp = (instances>0)
    tmp[split] = 0
    tmp = remove_small_objects(tmp, 100)
    tmp = label(tmp)[0]

    # re-assign labels in the removed parts
    _, indices = ndimage.distance_transform_edt(tmp==0, sampling=sampling, return_indices=True)
        
    outputs = (instances>0) * tmp[tuple(indices)]

    return outputs

            



    