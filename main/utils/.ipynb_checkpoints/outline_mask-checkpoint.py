import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes

def outline_mask(mask, iterations=5):
    
    mask = binary_fill_holes(mask>.5)
    outline = (~mask)*binary_dilation(mask, iterations=iterations)
    outline = outline.astype(np.float32)
    outline[outline==0] = np.NaN
    return outline