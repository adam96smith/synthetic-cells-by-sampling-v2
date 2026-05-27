import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

def instance_processing(mask, mode='Label', aniso_factor=1):
    
    '''
    Label the binary output of the segmentation model. 

    Mode: Watershed (for contacting cells) or Label (separated cells)
    '''

    if mode == 'Label':

        output = label(mask)

    elif mode == 'Watershed':
        dist_map = distance_transform_edt(mask==1, sampling=(aniso_factor, 1, 1))

        

    else:
        raise Exception('mode incorrectly assigned.')