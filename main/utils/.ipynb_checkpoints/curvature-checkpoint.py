import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import convolve, binary_dilation, binary_erosion


def circleKern(N, dims=3):
    ''' Generates a kernel with a circle/sphere fitted within '''
    if dims==2:
        assert N % 2 == 1
    
        X, Y = np.meshgrid(np.arange(N), np.arange(N))
    
        loc = [N//2, N//2]
    
        out = np.sqrt((X-loc[0])**2 + (Y-loc[1])**2 ) < N/2
        
    elif dims==3:
        assert N % 2 == 1
    
        X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    
        loc = [N//2, N//2, N//2]
    
        out = np.sqrt((X-loc[0])**2 + (Y-loc[1])**2 + (Z-loc[2])**2) < N/2

    else: 
        raise Exception('dims must be 2 or 3')

    return out


def circleKern_aniso(size, aniso_factor=1):
    ''' 
    Generates spherical kernel to fit specified anisotropic factor
    '''
    if aniso_factor == 1:
        return circleKern(size, dims=3)
    else:
        # ensure the centre of the kernel is included
        slices = list(np.arange(size//2, -50, -aniso_factor)[1:][::-1]) + list(np.arange(size//2, 50, aniso_factor))
        slices = np.array([x for x in slices if x in np.arange(size)])
        
        iso_kernel = circleKern(size, dims = 3)
        return iso_kernel[slices]
    

def outline_mask(mask, planar=False, mode='inner', iterations=1):
    assert mode in ['inner', 'outer']

    if planar:
        output = np.zeros_like(mask)
        for i in range(output.shape[0]):
            if mode == 'inner':
                output[i] = mask[i] * binary_dilation(~mask[i], iterations=iterations)
            else:
                output[i] = (~mask[i]) * binary_dilation(mask[i], iterations=iterations)
    else:
        if mode == 'inner':
            output = mask * binary_dilation(~mask, iterations=iterations)
        else:
            output = (~mask) * binary_dilation(mask, iterations=iterations)
            
    return output

def conv3d_speed_up(arr, kernel, device_prompt='cpu'):
    ''' GPU accelerated convolution for curvature calculation (~ x5 speedup, 1.5 seconds per mask) '''
    assert device_prompt in ['cpu', 'cuda']
    assert arr.ndim == 3
    assert kernel.ndim == 3

    if device_prompt=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # Convert input and kernel to Torch tensors
    arr_tensor = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, D, W, H)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, kD, kW, kH)

    # Compute convolution using PyTorch
    convolved_tensor = F.conv3d(arr_tensor, kernel_tensor, padding='same')

    # Move back to CPU and return as numpy array
    return convolved_tensor.squeeze().cpu().numpy()


def curvature_approximation_3d(mask, r=5, aniso_factor=1, planar=True):
    '''
    Approximate curvature on mask boundary by applying convolution with ball kernel.

    Step 1: A spherical (or circular) binary kernel is convolved with the input mask. 
            The resultant array is equivalent to the number of overlapping pixels 
            inside the ball, all interior pixels are set to zero.
    Step 2: Output describes the curvature at each point on the surface of the mask. 
            If the surface is flat, it will equal the half the number of pixels in the 
            spherical kernel, v_h (this includes the row/column in the centre). Concave 
            structures will result in greater values, whilst convex structures will 
            result in lesser values. The values are normalised to (v - v_h)/v_h, where 
            v is the value after convolution.Return normalised curvature values at mask 
            surface as an array

    Parameters:
        r:              Radius of the ball to convolve 
        aniso_factor:   Anisotropic factor of the data
        planar:         If the anisotropy is larger than r, apply planar convolution 
                        instead (optimises code)

    Note: If planar == False, we rescale the data to approximate the isotropic form.
          Otherwise, the normalisation breaks down. (can be GPU accelerated)
    '''

    assert mask.ndim == 3
    assert aniso_factor >= 1

    if r < aniso_factor:
        # Planar operatons should be used.
        planar = True

    # if planar apply all operations in the x,y - plane only
    if planar:
        # apply spherical convolution
        # curvature_array = convolve(mask.astype(float), circleKern(2*r+1, dims=2)[np.newaxis].astype(float))
        curvature_array = conv3d_speed_up(mask.astype(float), circleKern(2*r+1, dims=2)[np.newaxis].astype(float), device_prompt='cuda')
        outline = outline_mask(mask.astype(bool), mode='inner', iterations=1, planar=True)
        curvature_array *= outline

        # expected_value 
        v_h = np.sum(circleKern(2*r+1, dims=2)[:r+1])
    
        # normalise values
        curvature_array[curvature_array > 0] -= v_h
        curvature_array /= v_h
        
    else: # apply operations in all directions

        if aniso_factor > 1: # anisotropic kernel

            # ensure the centre of the kernel is included
            slices = list(np.arange((2*r+1)//2, -50, -aniso_factor)[1:][::-1]) + list(np.arange((2*r+1)//2, 50, aniso_factor))
            slices = np.array([x for x in slices if x in np.arange(2*r+1)])
            
            iso_kernel = circleKern(2*r+1, dims = 3)
            kernel = iso_kernel[slices]
        else:
            kernel = circleKern(r, dims=3)

        # curvature_array = convolve(mask.astype(float), kernel.astype(float))
        curvature_array = conv3d_speed_up(mask.astype(float), kernel.astype(float), device_prompt='cuda')
        outline = outline_mask(mask.astype(bool), mode='inner', iterations=1) ## inner for anisotropy
        curvature_array *= outline

        
        # expected_value 
        v_h = np.sum(kernel[:(kernel.shape[0]+1)//2])
    
        # normalise values
        curvature_array[curvature_array > 0] -= v_h
        curvature_array /= v_h

    return -curvature_array