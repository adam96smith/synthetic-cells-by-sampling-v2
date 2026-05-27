import numpy as np
from skimage.measure import label, regionprops



def jaccard_per_instance(pred, targ):
    """
    Compute Jaccard (IoU) and match scores per target instance efficiently.
    """

    assert pred.dtype == targ.dtype == int

    jaccard_data = {}
    match_data = {}

    # Handle empty prediction
    if pred.max() == 0:
        for i in np.unique(targ):
            if i > 0:
                jaccard_data[str(i)] = 0.0
                match_data[str(i)] = 0.0
        return jaccard_data, match_data

    # Flatten arrays for vectorized computation
    pred_flat = pred.ravel()
    targ_flat = targ.ravel()

    # Get valid pixels
    mask = (targ_flat > 0) | (pred_flat > 0)
    targ_flat = targ_flat[mask]
    pred_flat = pred_flat[mask]

    # Get unique labels
    targ_labels = np.unique(targ_flat)
    pred_labels = np.unique(pred_flat)
    targ_labels = targ_labels[targ_labels > 0]
    pred_labels = pred_labels[pred_labels > 0]

    # Compute intersections using joint labeling
    t = targ_flat.astype(np.int64); p = pred_flat.astype(np.int64)
    
    # ensure non-negative
    if (t < 0).any() or (p < 0).any():
        raise ValueError("targ_flat and pred_flat must be non-negative integer labels")
        
    # number of target and pred label slots (0..P-1) and (0..T-1)
    P = int(p.max()) + 1; T = int(t.max()) + 1 
    
    # encode pairs (use int64 to avoid overflow)
    combined = t * np.int64(P) + p
    
    # bincount with minlength ensures we can reshape to (T, P)
    counts = np.bincount(combined, minlength=T * P)
    
    # reshape to 2D contingency/intersection matrix:
    intersection = counts.reshape((T, P))
    # intersection[t, p] == number of pixels where targ==t and pred==p

    # Compute per-label areas
    area_t = np.bincount(targ_flat, minlength=T)
    area_p = np.bincount(pred_flat, minlength=pred_flat.max() + 1)

    # Compute IoU between every (targ, pred)
    for i in targ_labels:
        inter = intersection[i, pred_labels]
        union = area_t[i] + area_p[pred_labels] - inter
        valid = union > 0

        ious = np.zeros_like(pred_labels, dtype=float)
        ious[valid] = inter[valid] / union[valid]

        # Jaccard = max IoU for that target
        jaccard_data[str(i)] = ious.max() if ious.size > 0 else 0.0

        # Match score = intersection / target area
        matches = np.zeros_like(pred_labels, dtype=float)
        matches[valid] = inter[valid] / area_t[i]
        match_data[str(i)] = matches.max() if matches.size > 0 else 0.0

    return jaccard_data, match_data
















## Inefficient version

# def jaccard_per_instance(pred, targ):
#     '''
#     Jaccard Score and Match Score for each target instance
#     '''

#     assert pred.dtype == targ.dtype == int    

#     jaccard_data = {}
#     match_data = {}

#     if pred.max() == 0:
#         for i in np.unique(targ):
#             if i > 0:
#                 jaccard_data[str(i)] = 0
#                 match_data[str(i)] = 0

#     else:    
#         for i in np.unique(targ):
#             if i > 0:    
#                 jaccard_data[str(i)] = np.max([np.sum((targ==i)*(pred==j))/np.sum(np.maximum((targ==i),(pred==j))) for j in np.unique(pred) if j>0])
#                 match_data[str(i)] = np.max([np.sum((targ==i)*(pred==j))/np.sum(targ==i) for j in np.unique(pred) if j>0])
            

#     return jaccard_data, match_data

