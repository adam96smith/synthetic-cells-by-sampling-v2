import numpy as np
from scipy.ndimage import zoom, binary_fill_holes

def upscale_by_slice(arr, sf=2, xy_target=[512,512], order=1):
    '''
    If the training data is downsampled, the input image is downsampled the same. 
    Therefore, we need to upscale the output appropriately. 
    '''

    # assert arr.dtype == np.uint8

    xtarg,ytarg = xy_target
    output = np.zeros([arr.shape[0],xy_target[0],xy_target[1]], arr.dtype)

    if arr.dtype in [bool, np.uint8, np.uint16]:
        for lab in np.unique(arr):
            if lab > 0:
                # upscale each by slice
                for i in range(arr.shape[0]):
    
                    zoomed_img = zoom(arr[i] == lab, zoom=sf, order=order)
    
                    zoomed_img = binary_fill_holes(zoomed_img) # fill holes after 'zoom'
            
                    # xtmp, ytmp = zoomed_img.shape
                    xtmp, ytmp = np.minimum(zoomed_img.shape, xy_target)
    
                    # print((xtarg,ytarg),(xtmp,ytmp))
            
                    output[i,:xtmp,:ytmp] = np.maximum(output[i,:xtmp,:ytmp], lab*zoomed_img[:xtmp,:ytmp])

    else:
        for i in range(arr.shape[0]):
    
            zoomed_img = zoom(arr[i], zoom=sf, order=order)
    
            xtmp, ytmp = np.minimum(zoomed_img.shape, xy_target)
    
            output[i,:xtmp,:ytmp] = np.maximum(output[i,:xtmp,:ytmp], zoomed_img[:xtmp,:ytmp])

    return output