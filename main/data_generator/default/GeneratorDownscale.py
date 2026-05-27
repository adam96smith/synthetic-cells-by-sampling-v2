import glob
import h5py
import numpy as np
import os
from scipy.ndimage import gaussian_filter, convolve

'''
Functions for applying PSF (variable by slice).

We approximate a 3D PSF by performing the following operation:

    - Taking a maximum projection of the original image (along z).
    
    - Construct a blurred image by convolving the projection with a PSF that varies as a function of z:
        - A quadratic function is used to determine sigma: A*((z/a-1)**2 (A is max blur, a is slice with no blur)
        - Max. Value for A is 30, a should be around the middle slice +- 4

    - The image is too blurry to use, so use a weighted combo of the original image: p*(original) + (1-p)*(blurred)
        - p chosen between .75 and .9.

    - 
'''

def convolve_z_varying_psf(image3d, sig_list):
    """Convolve a 3D image with a Z-varying 3D PSF."""
    assert image3d.shape[0] == len(sig_list), "Z dimension mismatch"
    output = np.zeros_like(image3d, dtype=np.float32)
    
    for z, sig in zip(range(image3d.shape[0]), sig_list):
        psf = psf_efficient_2d(sigma = (sig,sig), tol=1e-6)
            
        output[z] = convolve(image3d.max(axis=0), psf)#[z]
    return output

def psf_efficient_2d(sigma=(1,1,1), tol=1e-3, anisotropy=1):
    '''
    Generate an optimal PSF kernel.
    tol: crop values less than value.
    '''

    res = 129
    
    psf = np.zeros([res,res])
    psf[res//2,res//2] = 1
    psf = gaussian_filter(psf, sigma=sigma)
    psf /= psf.sum()
    
    # Create mask for cropping based on threshold
    mask = psf > tol

    # Find bounding box of non-zero region
    x_any, y_any = np.any(mask, axis=(1)), np.any(mask, axis=(0))
    x_inds = np.where(x_any)[0]
    y_inds = np.where(y_any)[0]

    # Crop the PSF
    psf_cropped = psf[x_inds.min():x_inds.max()+1,
                      y_inds.min():y_inds.max()+1]

    psf_cropped /= psf_cropped.sum()

    return psf_cropped

'''
MAIN CODE
'''

image_files = sorted(glob.glob('synthetic_train/custom_texture/image_*.h5'))
mask_files = sorted(glob.glob('synthetic_train/mask_*.h5'))

os.makedirs('synthetic_train_ds/', exist_ok=True)
os.makedirs('synthetic_train_ds/custom_texture/', exist_ok=True)

generation_times = []
for i, image_file in enumerate(image_files):
    
    with h5py.File(image_file, 'r') as f:
        dataset = f['data']
        image = dataset[...]

    image = 1*image[:,:,::2,::2]

    ## apply PSF
    rng = np.random.rand(3)
    A = int(10+20*rng[0])
    a = image.shape[1]//2 + int(-4+8*rng[1])
    p = .75 + .15*rng[2]

    sig_list = A * ( (np.arange(image.shape[1])/a - 1)**2)
    
    blurred = convolve_z_varying_psf(image[0], sig_list)[np.newaxis]

    new_image = p*image + (1-p)*blurred
    
    with h5py.File(f'synthetic_train_ds/custom_texture/image_{str(i+1).zfill(5)}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=new_image.astype(np.float32))

    # Calculate time taken for this iteration 
    iteration_time = time.time() - start_time 
    generation_times.append(iteration_time) 
    
    if i % 10 == 0:
        # Calculate the average time taken so far 
        avg_time_per_iteration = sum(generation_times) / len(generation_times) 
        
        # Estimate remaining time 
        remaining_iterations = len(image_files) - (i + 1) 
        estimated_time_left = avg_time_per_iteration * remaining_iterations 
        
        # Convert estimated time to hours and minutes 
        hours_left = int(estimated_time_left // 3600) 
        minutes_left = int((estimated_time_left % 3600) // 60) 
        seconds_left = int(estimated_time_left % 60) 
        
        # Output estimated time remaining 
        print(f"{i+1}/{N_total}, Average Time per Cell: {avg_time_per_iteration}s, Estimated time left - {hours_left}h {minutes_left}m {seconds_left}s")



for i, mask_file in enumerate(mask_files):

    with h5py.File(mask_file, 'r') as f:
        dataset1 = f['data']
        mask = dataset1[...]

        dataset2 = f['instance']
        labels = dataset2[...]

    mask = 1*mask[:,:,::2,::2]
    labels = 1*labels[:,:,::2,::2]

    with h5py.File(f'synthetic_train_ds/mask_{str(i+1).zfill(5)}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=mask.astype(bool))
        dataset = hf.create_dataset('instance', data=labels.astype(int))