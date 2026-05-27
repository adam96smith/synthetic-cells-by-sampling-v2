import numpy as np
import h5py
from tifffile import imread
import glob
import re
import torch
from utils import custom_post_process, jaccard_per_instance
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def z_transform(img, mean=None, std=None):

    img = img.astype(np.float32)
    
    if mean is None:
        mean = img.mean()
        
    if std is None:
        std = img.std()

    return (img-mean) / std


def evaluate_during_training(model, 
                             image_list,
                             mask_list,
                             device, 
                             dataset_mean=None,
                             dataset_std=None,
                             downsample_factor=1,
                             dataset_id='default',
                             sampling = (1.0, 1.0, 1.0),
                            ):
    '''
    Evaluate model on a dataset during training.

    Supported formats:
     - CTC (GT volume or slice)
     - Elektronn3 Training Datasets

    Returns: mean_objective = mean Jaccard for all instances
    '''

    assert len(image_list) == len(mask_list)
    
    score_list = [];
    for i in tqdm(range(len(image_list)), desc='Evaluating Model'):

        image_file = image_list[i]
        mask_file = mask_list[i]

        if image_file.split('.')[-1] == 'tif':
            image = imread(image_file)[np.newaxis] # CTC (D x H x W) -> (1 x D x H x W)
            mask = imread(mask_file)[np.newaxis]
            
        elif image_file.split('.')[-1] == 'h5':
            with h5py.File(image_file, 'r') as f:
                image = f['data'][...]
            with h5py.File(mask_file, 'r') as f:
                mask = f['data'][...]
                
        else:
            raise Exception('Only .tif and .h5 files supported.')

        
       

        if downsample_factor > 1:
            if mask.ndim == 4:
                mask = mask[:,:,::downsample_factor,::downsample_factor]
            else:
                mask = mask[:,::downsample_factor,::downsample_factor]            
            image = image[:,:,::downsample_factor,::downsample_factor]
            
        image = image.astype(np.float32)
        mask = mask.astype(int) 
        
        # Normalise Images            
        if dataset_mean is None or dataset_std is None:
            image = z_transform(image, mean=image.mean(), std=image.std())
        else:
            image = z_transform(image, mean=dataset_mean, std=dataset_std)

        # print(f'Real Image: Mean - {image.mean():.4f}, Std. - {image.std():.4f}.')
        
        # Run model
        inp = torch.from_numpy(image[np.newaxis]).float().to(device)
        with torch.no_grad():  # Save memory by disabling gradients
            output = model(inp).cpu()
        probs = output.softmax(1).detach().numpy()

        prediction = np.array(np.argmax(probs, axis=1))
        try:
            prediction = custom_post_process(prediction, mode = dataset_id)
        except:
            print('Raw Model Output')
            prediction = label(prediction)
        prediction = prediction.astype(int)

        ''' 
        If multi-cell image, use voronoi to split them. 
        We are really only interested in how accurate 
        the segmentation is, so using Voronoi we can 
        avoid instance merging if that occurs !!
        '''

        if image.ndim == mask.ndim: # Volume Labels

            if len(np.unique(mask)) > 2: # BG + 1 label > 0
                # use voronoi of GT to separate semantic segmentation
                _, indices = distance_transform_edt(mask[0]==0, 
                                                    sampling=sampling, 
                                                    return_indices=True)
                voronoi = 1*mask[0][tuple(indices)].astype(int)
                prediction *= voronoi[np.newaxis]
            
            jaccard_data, _ = jaccard_per_instance(prediction, mask)
            
        else: # Slice Labels
            zs = int(re.findall(r'\d+', mask_file)[-1])
            pred = 1*prediction[:,zs]

            if len(np.unique(mask)) > 2: # BG + 1 label > 0
                # use voronoi of GT to separate semantic segmentation
                _, indices = distance_transform_edt(mask[0]==0, 
                                                    sampling=sampling[1:], 
                                                    return_indices=True)
                voronoi = 1*mask[0][tuple(indices)].astype(int)
                pred *= voronoi[np.newaxis]
                
            jaccard_data, _ = jaccard_per_instance(pred, mask)

        score_list += [jaccard_data[s] for s in jaccard_data]

    # JaccardScore
    if len(score_list) == 0: # no matches
        mean_objective = 0
    else:
        mean_objective = np.mean(score_list)

    return mean_objective