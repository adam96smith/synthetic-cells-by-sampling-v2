# import numpy as np
# from scipy.ndimage import binary_fill_holes

# def fill_holes_by_slice(mask):
#     """
#     Fills enclosed gaps in each z-slice of a 3D binary mask along the x-y plane.

#     Args:
#         mask (numpy.ndarray): A 3D binary array (z, x, y).

#     Returns:
#         numpy.ndarray: A 3D binary array with holes filled in each z-slice.
#     """
#     filled_mask = np.zeros_like(mask, dtype=bool)

#     for z in range(mask.shape[0]):  # Iterate through each z-slice
#         filled_mask[z] = binary_fill_holes(mask[z])  # Fill holes in x-y plane

#     return filled_mask

import numpy as np
from scipy.ndimage import binary_fill_holes

def fill_holes_by_slice(mask):
    """
    Fills enclosed gaps in each z-slice of a 3D mask along the x-y plane.

    Args:
        mask (numpy.ndarray): A 3D array (z, x, y).
                             - If binary, fills holes per slice.
                             - If labeled (integers >0), fills holes per label per slice.

    Returns:
        numpy.ndarray: A 3D array with holes filled.
    """
    # Assert: only boolean or integer labels allowed
    assert np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.integer), \
        "mask must be a binary (bool or 0/1) array or an integer-labeled array."

    # Case 1: Binary mask
    if mask.dtype == bool or np.array_equal(np.unique(mask), [0, 1]):
        filled_mask = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[0]):
            padded1 = np.pad(mask[z], ((1,1),(0,0)), constant_values=True)
            padded2 = np.pad(mask[z], ((0,0),(1,1)), constant_values=True)
    
            filled1 = binary_fill_holes(padded1)[1:-1]
            filled2 = binary_fill_holes(padded2)[:,1:-1]
    
            filled_mask[z] = np.maximum(filled1, filled2)
        return filled_mask.astype(mask.dtype)

    # Case 2: Label mask
    else:
        filled_mask = np.copy(mask)
        labels = np.unique(mask)
        labels = labels[labels != 0]  # skip background

        for lbl in labels:
            for z in range(mask.shape[0]):
                slice_mask = (mask[z] == lbl)
                if slice_mask.any():
                    padded1 = np.pad(slice_mask, ((1,1),(0,0)), constant_values=True)
                    padded2 = np.pad(slice_mask, ((0,0),(1,1)), constant_values=True)
            
                    filled1 = binary_fill_holes(padded1)[1:-1]
                    filled2 = binary_fill_holes(padded2)[:,1:-1]
            
                    filled_slice = np.maximum(filled1, filled2)
                    
                    # Assign filled-in pixels back to label
                    filled_mask[z][filled_slice] = lbl

        return filled_mask