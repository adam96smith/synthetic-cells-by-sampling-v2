'''
Selected Dataset Quality Measures from CTC

Het_i - Heterogeneity of signal inside the cells
Het_b - Heterogeneity of signal between the cells

custom-Het_i - values at the inward normal of mesh rather than full values
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import map_coordinates

def het_i_calc(image, annotations):
    '''
    Input a single image frame with annotations.

    Output, list of internal Het_i measures for each annotation.
    '''

    assert image.shape == annotations.shape
    
    output = []

    n = annotations.max()
    if n > 0:

        BG = np.mean(image[annotations==0])

        for lab in np.unique(annotations):
            if lab != 0:
                if np.sum(annotations==lab) > 0:
                    FG_std = np.std(image[annotations==lab])
                    FG_mean = np.mean(image[annotations==lab])
                    
                    Y = FG_std / np.abs(FG_mean - BG)
        
                    output.append(Y)

    return output

    
def het_b_calc(image, annotations):
    '''
    Input a single image frame with annotations.

    Output, list of internal Het_i measures for each annotation.
    '''

    assert image.shape == annotations.shape
    
    Y = []

    n = annotations.max()
    output = []
    if n > 1: # must have more than 1 label to calculate

        BG = np.mean(image[annotations==0])

        for lab in np.unique(annotations):
            if lab != 0:

                FG_mean = np.mean(image[annotations==lab])
    
                Y.append(FG_mean - BG)

        Y = np.array(Y)

        output += list(Y / (np.sum( np.abs(Y) )/(len(np.unique(annotations)))-1) )

    return output


''' CUSTOM '''

def mesh_with_values(voronoi, values, spacing=(1,1,1), max_distance=10.0, operation='max', mode='inner', step=0.5):
    """
    Extract a mesh from a 3D binary mask and compute the value 
    along inward normals at each vertex. 

    Spacing is voxel spacing of the data. binary and all arrays 
    in value must have the same dimension

    The max_distance, and Min or Max is computed for each values.
    All the same if not a list. 
    
    Process is applied to all arrays in values
    """
    
    # Normalize list inputs
    if not isinstance(values, list):
        values = [values]

    def to_list(param, n):
        return param if isinstance(param, list) else [param]*n

    distance_list = to_list(max_distance, len(values))
    operation_list = to_list(operation, len(values))
    mode_list = to_list(mode, len(values))

    for m in mode_list:
        assert m in ['inner', 'outer', 'both']
        
    '''
    Process arrays
    '''
    
    # Extract mesh
    verts, faces, normals, _ = measure.marching_cubes(voronoi==1, level=0.5, spacing=spacing)
        
    extracted_values = [np.zeros(len(verts), dtype=float) for _ in values]   
    
    for i, (v, n) in enumerate(zip(verts, normals)): # flipped normals
        for X, max_distance, op, m, Y in zip(values, distance_list, operation_list, mode_list, extracted_values):
            # points along normal
            if m == 'inner':
                distances = -np.arange(0, max_distance, step)
            elif m == 'outer':
                distances = np.arange(0, max_distance, step)
            else:
                distances = np.arange(-max_distance, max_distance, step)
                
            points = v + np.outer(distances, n)  # shape (L,3)
            
            # convert to voxel coords (z,x,y indexing, spacing already applied)
            voxel_coords = (points / spacing).T  # (3,L) for map_coordinates
            
            # sample intensities
            vals = map_coordinates(X, voxel_coords, order=1, mode="nearest")
            
            # take maximum
            if op == 'max':
                Y[i] = vals.max()
            elif op == 'mean':
                Y[i] = vals.mean()
            else:
                Y[i] = vals.min()
            
    
    return verts, faces, extracted_values


def contour_with_values(binary, values, spacing=(1,1), max_distance=10.0, 
                        operation='max', mode='inner', step=0.5):
    """
    Extract a mesh from a 2D binary mask and compute the value 
    along the normals at each contour vertex. 

    Spacing is voxel spacing of the data. binary and all arrays 
    in value must have the same dimension

    The max_distance, and Min or Max is computed for each values.
    All the same if not a list. 
    
    Process is applied to all arrays in values
    """
    # Normalize list inputs
    if not isinstance(values, list):
        values = [values]

    def to_list(param, n):
        return param if isinstance(param, list) else [param]*n

    distance_list = to_list(max_distance, len(values))
    operation_list = to_list(operation, len(values))
    mode_list = to_list(mode, len(values))

    # Extract contours using marching squares
    contours = measure.find_contours(binary.astype(float), level=0.5)

    results = []
    for contour in contours:
        # Scale coordinates by spacing
        contour_scaled = np.zeros_like(contour)
        contour_scaled[:, 0] = contour[:, 0] * spacing[0]
        contour_scaled[:, 1] = contour[:, 1] * spacing[1]

        # # Compute tangent vectors and normals
        # diffs = np.gradient(contour_scaled, axis=0)
        # tangents = np.stack([diffs[:,1], -diffs[:,0]], axis=1)
        # tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
        # # Outward normals (rotate tangents 90°)
        # normals = np.stack([-tangents[:,1], tangents[:,0]], axis=1)
        
        # Compute tangent vectors and normals
        diffs = np.gradient(contour_scaled, axis=0)
        normals = np.stack([diffs[:,1], -diffs[:,0]], axis=1)
        normals /= -np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

        extracted_values = [np.zeros(len(contour_scaled), dtype=float) for _ in values]

        for i, (v, n) in enumerate(zip(contour_scaled, normals)):
            for X, dmax, op, m, Y in zip(values, distance_list, operation_list, mode_list, extracted_values):
                if m == 'inner':
                    distances = np.arange(0, dmax, step)
                elif m == 'outer':
                    distances = -np.arange(0, dmax, step)
                else:
                    distances = np.arange(-dmax, dmax, step)

                # Points along normal (in physical coords)
                points = v + np.outer(distances, n)

                # Convert to pixel coords (y,x)
                pixel_coords = (points / spacing).T  # (2, L)
                vals = map_coordinates(X, pixel_coords, order=1, mode='nearest')

                if op == 'max':
                    Y[i] = vals.max()
                elif op == 'mean':
                    Y[i] = vals.mean()
                else:
                    Y[i] = vals.min()

        # results.append(())

    return contour_scaled, normals, extracted_values


def custom_het_i_calc(image, annotations, spacing=(1.0,1.0,1.0), max_depth=1.0, operation='max'):
    '''
    Input a single image frame with annotations.

    Output, list of internal Het_i measures for each annotation.
    '''

    assert image.shape == annotations.shape
    
    output = []

    n = annotations.max()
    if n > 0:

        BG = np.mean(image[annotations==0])

        for lab in np.unique(annotations):
            if lab != 0:
                if np.sum(annotations==lab) > 0:

                    if image.ndim == 3:
                        _, _, vals = mesh_with_values(annotations==lab, [image],
                                                      spacing=spacing, 
                                                      max_distance=max_depth,
                                                      operation=operation, 
                                                      mode='inner')
                    elif image.ndim == 2:
                        _, _, vals = contour_with_values(annotations==lab, [image],
                                                         spacing=spacing[1:], 
                                                         max_distance=max_depth,
                                                         operation=operation, 
                                                         mode='inner')
                    
                    FG_std = np.std(vals)
                    FG_mean = np.mean(vals)
                    
                    Y = FG_std / np.abs(FG_mean - BG)
        
                    output.append(Y)

    return output