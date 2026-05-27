import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

'''

Functions for applying 3D PSF to image.

Assumption: 

'''

def psf_aniso(sigma=(1,1,1), anisotropy=1, z_scale=1):
    '''
    Generate anisotropic PSF kernel.
    '''

    res = 65
    
    psf = np.zeros([int(z_scale)*res,res,res])
    psf[int(z_scale*res)//2,res//2,res//2] = 1
    psf = gaussian_filter(psf, sigma=sigma)
    psf = psf[::anisotropy]
    psf /= psf.sum()
    
    return psf

def psf_aniso_efficient(sigma=(1,1,1), tol=1e-6, anisotropy=1):
    '''
    Generate an optimal PSF kernel.
    tol: crop values less than value.
    '''

    res = 257
    
    psf = np.zeros([res,res,res])
    psf[res//2,res//2,res//2] = 1
    psf = gaussian_filter(psf, sigma=sigma)
    psf = psf[::anisotropy]
    psf /= psf.sum()
    
    # Create mask for cropping based on threshold

    # if all values are less than tol, keep full kernel
    if np.sum(psf > tol) > 0:
        mask = psf > tol

        # Find bounding box of non-zero region
        z_any, x_any, y_any = np.any(mask, axis=(1,2)), np.any(mask, axis=(0,2)), np.any(mask, axis=(0,1))
        z_inds = np.where(z_any)[0]
        x_inds = np.where(x_any)[0]
        y_inds = np.where(y_any)[0]
    
        # Crop the PSF
        psf_cropped = psf[z_inds.min():z_inds.max()+1,
                          x_inds.min():x_inds.max()+1,
                          y_inds.min():y_inds.max()+1]
    
        psf_cropped /= psf_cropped.sum()
    else:
        psf_cropped = psf

    return psf_cropped
    

def apply_z_varying_psf(img, sig_list, anisotropy=1, z_scale=2):
    """
    Convolve a 3D image with a Z-varying 3D PSF.

    img: 3D image
    sig_list: List of sigma as a function of z
    anisotropy: Anisotropy of img
    z_scale: Scaling factor for PSF in z-direction 
             (to exaggerate blur in z).
    
    """
    assert img.shape[0] == len(sig_list), "Z dimension mismatch"
    output = np.zeros_like(img, dtype=np.float32)
    
    for z, sig in zip(range(img.shape[0]), sig_list):
        psf = psf_aniso(sigma = (z_scale*sig,sig,sig), anisotropy=anisotropy, z_scale=z_scale)
        output[z] = fftconvolve(img, psf, mode='same')[z]
    return output