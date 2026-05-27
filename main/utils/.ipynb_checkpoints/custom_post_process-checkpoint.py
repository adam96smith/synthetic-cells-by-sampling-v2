import numpy as np
from skimage.morphology import isotropic_closing
from utils.fill_holes_by_slice import fill_holes_by_slice # might not work
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label
from scipy.ndimage import binary_dilation, binary_erosion, center_of_mass, generate_binary_structure

from skimage.segmentation import watershed
from scipy.ndimage import affine_transform
from skimage.feature import peak_local_max

from utils.curvature import *
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from scipy import stats

def rotate_image(image, angle, center=None, order=1, mode='constant', cval=0.0):
    """
    Rotate a 2D image by `angle` radians around a given center point.
    """
    h, w = image.shape
    if center is None:
        center = (h / 2.0, w / 2.0)

    cy, cx = center

    # rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    # build full affine matrix (2x3 for scipy)
    # shift center to origin, rotate, then shift back
    offset = np.array([cy, cx]) - R @ np.array([cy, cx])

    rotated = affine_transform(
        image,
        R,
        offset=offset,
        order=order,
        mode=mode,
        cval=cval
    )
    return rotated

def get_min_cross_section(binary):
    '''
    Get the minimum bounding box width of binary shape using rotations
    '''

    if binary.max() == 0:
        return 1000

    lab = label(binary)[0]
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    mask = lab == largest_label

    binary_center = center_of_mass(mask)
    
    min_width = 1000
    for rotation in np.arange(0,2*np.pi, np.pi/16):
    
        mask_r = rotate_image(mask, rotation, binary_center)
    
        proj = np.arange(mask.shape[1])[mask_r.max(axis=0)]
    
        min_width = min(proj.max() - proj.min(), min_width)
        
    return min_width

def reassign_small_labels(labeled, min_size=50, max_distance=np.inf):
    """
    Reassign labels smaller than a threshold to the nearest large label
    if they are within a distance threshold.

    Args:
        labeled (ndarray): 3D labeled array (ints, background=0).
        min_size (int): Minimum size for keeping a label.
        max_distance (float): Maximum distance allowed for reassignment.

    Returns:
        ndarray: New labeled array with small labels reassigned.
    """
    labeled = labeled.copy()
    labels, counts = np.unique(labeled, return_counts=True)
    label_sizes = dict(zip(labels, counts))

    # Identify small and large labels
    small_labels = {lab for lab, size in label_sizes.items() if size < min_size and lab != 0}
    large_labels = {lab for lab, size in label_sizes.items() if size >= min_size and lab != 0}

    if not small_labels:
        return labeled  # Nothing to do

    # Create mask of "large" objects
    large_mask = np.isin(labeled, list(large_labels))

    # Compute nearest large voxel and distance map
    distances, nearest = distance_transform_edt(~large_mask, return_indices=True)
    nearest_labels = labeled[tuple(nearest)]

    for lab in small_labels:
        mask = labeled == lab

        # Minimum distance from this object to a large object
        min_dist = distances[mask].min()

        if min_dist <= max_distance:
            # Reassign *all* pixels of this object
            nearest_majority = np.bincount(nearest_labels[mask]).argmax()
            labeled[mask] = nearest_majority
        else:
            # Leave the small label intact
            pass

    return labeled


def label_with_boundary_merging(binary, distance_thresh):
    """
    Label binary regions, merging disconnected components if their 
    boundaries are within a specified distance.
    
    Returns
    -------
    merged_labels : ndarray of int
        Labeled image with merged regions.
    """
    # Step 1: initial labeling
    init_labels = label(binary)[0]
    props = regionprops(init_labels)
    n = len(props)
    if n == 0:
        return np.zeros_like(binary, dtype=int)
    if n == 1:
        return init_labels
    
    # Step 2: extract boundary coordinates for each component
    struct = generate_binary_structure(binary.ndim, connectivity=1)
    boundaries = []
    for p in props:
        mask = (init_labels == p.label)
        eroded = binary_erosion(mask, struct, border_value=0)
        boundary = mask ^ eroded  # XOR to get boundary voxels
        coords = np.argwhere(boundary)
        boundaries.append(coords)
    
    # Step 3: compute min distances between boundaries
    adjacency = np.eye(n, dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            d = np.min(cdist(boundaries[i], boundaries[j]))
            if d <= distance_thresh:
                adjacency[i, j] = adjacency[j, i] = 1
    
    # Step 4: merge using connected components in adjacency graph
    graph = csr_matrix(adjacency)
    n_groups, groups = connected_components(graph, directed=False)
    
    # Step 5: relabel
    merged_labels = np.zeros_like(init_labels, dtype=int)
    for idx, p in enumerate(props):
        merged_labels[init_labels == p.label] = groups[idx] + 1
    
    return merged_labels

def relabel_by_axis(label_img, axis=1):
    """
    Relabel objects in a 3D labelled image so that their IDs are ordered
    by the centroid position along a chosen axis.

    Parameters
    ----------
    label_img : ndarray, int
        3D labelled image (0 = background, >0 = object labels).
    axis : int
        Axis to order by (0=z, 1=x, 2=y).

    Returns
    -------
    new_labels : ndarray, int
        Relabelled image with objects numbered according to position.
    mapping : dict
        Mapping {old_label: new_label}.
    """
    labels = np.unique(label_img)
    labels = labels[labels > 0]  # exclude background

    if len(labels) == 0:
        return label_img.copy(), {}

    # compute centroids
    centroids = {}
    for lbl in labels:
        coords = np.argwhere(label_img == lbl)
        centroids[lbl] = coords[:, axis].mean()

    # sort by centroid position along the axis
    sorted_labels = sorted(labels, key=lambda l: centroids[l])

    # build mapping
    mapping = {old: new for new, old in enumerate(sorted_labels, start=1)}

    # relabel image
    new_img = np.zeros_like(label_img, dtype=int)
    for old, new in mapping.items():
        new_img[label_img == old] = new

    return new_img, mapping


def z_inherit(seg):
    '''
    If the labelling is greater than 2 then it is because the cells are split in the z-direction.

    For all labels greater than 2, they are to inherit the most common label within the z-direction
    '''

    assert seg.ndim == 4
    
    fixed_seg = np.zeros_like(seg)
    for t in range(seg.shape[0]):
        
        if seg[t].max() <= 2: # fix not required for this frame
            fixed_seg[t] = 1*seg[t]

        else:
            fixed_seg[t] = 1*(seg[t] * (seg[t]<=2))
            for v in range(3, seg[t].max()+1):

                x = np.max(seg[t]==v, axis=0) # footprint of mislabelling

                # mode seg label (<=2) in the z-direction
                tmp = (seg[t] * (seg[t]<=2)).astype(np.float32)
                tmp[tmp==0] = np.NaN
                mode_proj = stats.mode(tmp, nan_policy='omit', axis=0)[0]#
                mode_proj[np.isnan(mode_proj)] = 0

                # mode non-zero val within the footprint
                val = stats.mode(mode_proj[x | (mode_proj>0)])[0]

                # reassign value
                fixed_seg[t][seg[t] == v] = val

            # dilate and erode to remedy the separation artefact
            for v in range(1, fixed_seg[t].max()+1):
                '''
                Note: caution about erosion artefact at the boundary 
                '''
                tmp = binary_dilation(fixed_seg[t]==v)
                tmp = binary_erosion(tmp)

                fixed_seg[t][tmp] = v # should be fine!

    return fixed_seg


def keep_largest_component(mask, connectivity=1):
    """
    Keep only the largest connected component in a binary array.
    """
    
    labeled, num = label(mask)

    if num == 0:
        return mask.astype(bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background

    largest_label = counts.argmax()

    return labeled == largest_label
    
''' MAIN '''

def custom_post_process(prediction, mode='default'):
    '''
    All Post-Processing Steps for A549 Motile Cancer Cell Predicitons
    '''

    assert prediction.ndim == 4 

    if mode == 'default':
        output = prediction # just return raw model output

    elif mode == 'TOY':
        
        # Standard Post-Processing 
        output = binary_fill_holes(prediction[0]) # just return raw model output
        output = remove_small_objects(output, 50)
        output = label(output)[0][np.newaxis]

    elif mode in ['A549','A549-SIM']:

        filled = fill_holes_by_slice(prediction[0] > .5)
        filled = binary_fill_holes(filled)

        # remove objects 5 mu from largest connected region
        labelled_output = label(filled)[0]
        N = np.argmax(np.bincount(labelled_output.flatten())[1:]) + 1
        dist_map = distance_transform_edt(~(labelled_output == N), sampling=(1, .126, .126))

        output = (filled * (dist_map<5))[np.newaxis]

    elif mode == 'A549_OLD':
        ''' Old Method for A549 Dataset '''
        filled = fill_holes_by_slice(prediction[0] > .5)

        # remove objects 5 mu from largest connected region
        labelled_output = label(filled)[0]
        N = np.argmax(np.bincount(labelled_output.flatten())[1:]) + 1
        dist_map = distance_transform_edt(~(labelled_output == N), sampling=(1, .126, .126))

        output = (filled * (dist_map<5))[np.newaxis]

    elif mode == 'CE':
        
        """
        Dilates labeled regions while keeping integer labels by assigning each new pixel
        the label of the nearest original labeled pixel.
        
        Parameters
        ----------
        label_img : np.ndarray
            Integer label image (0 = background, >0 = object IDs).
        dilation_radius : int
            Radius (in pixels) for the dilation.
        
        Returns
        -------
        np.ndarray
            Dilated label image with preserved integer IDs.
        """
        label_img = 1*prediction[0] # stardist prediction is labelled
    
        # remove small objects in each z-slice
        label_cleaned = np.zeros_like(label_img)
        for z in range(label_img.shape[0]):
            label_cleaned[z] = remove_small_objects(label_img[z], 500)
    
        # Boolean mask of where the objects are
        mask = label_cleaned > 0
        
        # Dilate 
        dilated_mask = binary_dilation(mask, np.array([1,1,1],bool)[:,np.newaxis,np.newaxis], iterations=2)
        dilated_mask = binary_dilation(dilated_mask, 
                                       np.array([[0,1,0],
                                                 [1,1,1],
                                                 [0,1,0]],bool)[np.newaxis], 
                                       iterations=1)
        
        # Distance transform with nearest original pixel index
        dist, nearest_idx = distance_transform_edt(~mask, return_indices=True)
        
        # Create output copy
        output = np.zeros_like(label_cleaned)
        
        # For all pixels in the dilated mask, assign nearest label
        nearest_labels = label_cleaned[tuple(nearest_idx)]
        output[dilated_mask] = nearest_labels[dilated_mask]
        output = output[np.newaxis]

    elif mode == 'CHO':
        
        """
        CHO Post-process 3D segmentation with classes:
          0 = background, 1 = interior, 2 = edge
    
        Steps:
          - Fill holes in class==1 per XY slice
          - Label connected components of class==1 in 3D
          - Assign each class==2 voxel to nearest class==1 label in 3D
          - Dilate/Erode output to correct output
    
        Args:
            segmentation (ndarray): 3D numpy array (Z, X, Y)
    
        Returns:
            ndarray: 3D labeled instance mask (integers, background=0)
        """
        # Step 1: Fill holes in each slice
        filled = fill_holes_by_slice(prediction[0]==1)
    
        # Step 2: Label connected components in 3D
        labeled, n_labels = label(filled)
        labeled = reassign_small_labels(labeled, min_size=1000, max_distance=10)
    
        # Step 3: Assign edges to nearest interior label
        edges = prediction[0] == 2
        if edges.any() and labeled.max() > 0:
            # Distance to nearest labeled voxel
            _, nearest = distance_transform_edt(labeled == 0, return_indices=True)
            nearest_labels = labeled[tuple(nearest)]
            labeled[edges] = nearest_labels[edges]
    
        # Step 4: Dilate/Erode output to correct output
        output = np.zeros_like(labeled)[np.newaxis]
        iters = 4
        for i in np.unique(labeled):
            if i > 0:
                tmp = binary_dilation(labeled==i)
                tmp = fill_holes_by_slice(tmp)
                tmp = np.pad(tmp, ((0,0),(iters,iters),(iters,iters)), mode='edge')
                tmp = binary_erosion(tmp, np.array([[0,1,0],[1,1,1],[0,1,0]], bool)[np.newaxis], iterations=iters)
                tmp = tmp[:,iters:-iters,iters:-iters]
                output = np.maximum(output, i*tmp[np.newaxis])

    elif mode in ['SHAINY1', 'SHAINY2', 'cell_doublets']:

        # ## fill holes in cell seg, erode, get bounding box width
        # filled_seg = fill_holes_by_slice(prediction[0]!=0)
        # # filled_seg = largest_connected_component(filled_seg) # cells might not be touching!
        # filled_seg = remove_small_objects(filled_seg, 5000)
        
        # kern_3 = np.array([[0,1,0],[1,1,1],[0,1,0]], bool)
        # kern_5 = np.array([[[0,0,0,0,0],
        #                     [0,0,1,0,0],
        #                     [0,1,1,1,0],
        #                     [0,0,1,0,0],
        #                     [0,0,0,0,0]],
        #                    [[0,1,1,1,0],
        #                     [1,1,1,1,1],
        #                     [1,1,1,1,1],
        #                     [1,1,1,1,1],
        #                     [0,1,1,1,0]],
        #                    [[0,0,0,0,0],
        #                     [0,0,1,0,0],
        #                     [0,1,1,1,0],
        #                     [0,0,1,0,0],
        #                     [0,0,0,0,0]]], bool)
        # filled_seg = binary_erosion(filled_seg, kern_3[np.newaxis], iterations=2)
        
        # r = get_min_cross_section(filled_seg.max(axis=0))
        # r_c = r//2
        
        # ## calculate curvature
        # if filled_seg.max() > 0:
        #     curvature = curvature_approximation_3d(filled_seg, r=r_c, aniso_factor=2, planar=False)
        # else:
        #     return np.zeros(prediction.shape, bool)
        
        # ## mask negative curvature
        # filled_curvature = np.zeros(curvature.shape, bool)

        # curvature_labels = label_with_boundary_merging(curvature<0, r//8)
        # # curvature_labels = 1*remove_small_objects((curvature<0), 15)#.astype(int)
        
        # for R in regionprops(curvature_labels):
        #     z0,x0,y0,z1,x1,y1 = R.bbox
        
        #     if np.sum(R.image) > 10:
        #         filled_curvature[z0:z1,x0:x1,y0:y1] = R.image_convex
        
        # output = label(filled_seg*(~filled_curvature))[0]
        # output = reassign_small_labels(output, min_size=200)
        # counter = 1
        # while output.max() < 2 and counter < 2*r//3: # stopping condition
        #     '''
        #     Filled Curvature should separate out the cells across the interface.
        #     Sometimes the cells are still connected, so dilating the filled_curvature
        #     can solve this. Applying the dilation too much can ruin the method,
        #     so cap at 10 iteration. If cells cannot be split, end the process.
        #     '''
        #     filled_curvature = binary_dilation(filled_curvature, kern_5)
        #     output = label(filled_seg*(~filled_curvature))[0]        
        #     output = reassign_small_labels(output, min_size=200)
        #     counter += 1

        # if counter == 2*r//3:
        #     print('Cells could not be split')
        #     return output[np.newaxis]
            
        # # assign filled_curvature voxels to nearest label
        # _, indices = distance_transform_edt(output==0, sampling = (2,1,1), return_indices=True)
            
        # output = output[tuple(indices)]
        # output *= filled_seg
        # output *= remove_small_objects(output > 0, 5000)
        # output = reassign_small_labels(output, min_size=200)
        # output = relabel_by_axis(output)[0][np.newaxis]
        # output = z_inherit(output)

        output = fill_holes_by_slice(prediction[0] > .5).astype(bool)
        output = remove_small_objects(output, 3000)[np.newaxis]

    elif mode == 'DICTY':
        
        output = keep_largest_component(prediction.astype(bool))

        iterations = 1
        kern = circleKern(5)[::2]
        output = binary_fill_holes(output[0]) # just return raw model output
        
        output = binary_dilation(output, kern, 
                                 iterations=iterations)
        output = binary_fill_holes(output)
        output = binary_erosion(output, kern, 
                                iterations=iterations)[np.newaxis]

        output = output.astype(int)

    elif mode == 'BADEER':
        
        output = binary_fill_holes(prediction.astype(bool)[0])
        output = remove_small_objects(output, 1000)
        output = (label(output)[0][np.newaxis]).astype(int)

    else:
        raise Exception('mode needs to be added to utils.custom_post_process')

    return output