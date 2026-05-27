import numpy as np

def z_transform(img, mean=None, std=None):

    img = img.astype(np.float32)
    
    if mean is None:
        mean = img.mean()
        
    if std is None:
        std = img.std()

    return (img-mean) / std