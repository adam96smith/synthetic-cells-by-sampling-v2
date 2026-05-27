import numpy as np

def jitter_partition_map(partitions, labels, sigma):
    """
    Jitter a partition map by sampling random voxel-wise displacements
    from Gaussian(s). Inside-cell voxels remain fixed.
    
    Parameters
    ----------
    partitions: ndarray, shape (Z,X,Y)
        Integer label map for nearest cell.
    labels : ndarray, shape (Z,X,Y)
        Integer label map (0=background, 1+=cell labels).
    sigma : float or sequence of 3 floats
        Standard deviation(s) for displacement sampling (in voxels).
        - float: isotropic jitter
        - (sigma_z, sigma_x, sigma_y): anisotropic jitter
    rng : np.random.Generator
        Random number generator.
        
    Returns
    -------
    jittered : ndarray, shape (Z,X,Y)
        Partition map with jittered background/partition boundaries.
    """
    
    assert partitions.shape == labels.shape
    
    Z, X, Y = partitions.shape
    
    # normalize sigma input
    if np.isscalar(sigma):
        sigma_z, sigma_x, sigma_y = sigma, sigma, sigma
    else:
        if len(sigma) != 3:
            raise ValueError("sigma must be a float or a sequence of 3 floats (z, x, y)")
        sigma_z, sigma_x, sigma_y = sigma

    # coordinate grid
    zz, xx, yy = np.meshgrid(np.arange(Z), np.arange(X), np.arange(Y),
                             indexing="ij", sparse=False)
    
    # random displacements (rounded to nearest int)
    dz = np.random.normal(0, sigma_z, size=partitions.shape).round().astype(int)
    dx = np.random.normal(0, sigma_x, size=partitions.shape).round().astype(int)
    dy = np.random.normal(0, sigma_y, size=partitions.shape).round().astype(int)
    
    # keep interior of cells fixed
    mask_interior = labels > 0
    dz[mask_interior] = 0
    dx[mask_interior] = 0
    dy[mask_interior] = 0
    
    # displaced coordinates, clamped
    z_new = np.clip(zz + dz, 0, Z-1)
    x_new = np.clip(xx + dx, 0, X-1)
    y_new = np.clip(yy + dy, 0, Y-1)
    
    # gather reassigned partitions
    jittered = partitions[z_new, x_new, y_new]
    
    return jittered
