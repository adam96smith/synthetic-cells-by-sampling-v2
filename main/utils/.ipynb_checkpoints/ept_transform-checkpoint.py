
import numpy as np
from scipy.ndimage import map_coordinates
from utils.perlin_noise import perlin_optim_3d


# Custom Elastic Deformation
class ElasticPerlinTransform:
    '''
    Elastic Deformation using Perlin noise to generate smooth distortion.

    Inspired by 
    Elastic deformation of images as described in Simard, 2003
    (DOI 10.1109/ICDAR.2003.1227801)

    ** Currently only supports 3D Images **

    Args:
        prob: probability (between 0 and 1) with which to perform this
            augmentation. The input is returned unmodified with a probability
            of ``1 - prob``
        grid_points: 
        p: distortion amplitude as a fraction of resolution

    Important:
        Since the amplitude of the change is relative to the input 
        resolution, anisotropic factor is accounted for.
        
    
    '''
    def __init__(self,
                 prob: float = 0.25,
                 grid_points: int = 8,
                 p: float = .05,
                 sf: int = 2,
                 order: int = 3,
    ):        
        self.prob = prob
        self.grid_points = grid_points
        self.p = p
        self.sf = sf # optimisation parameter
        self.order = order
        
    def __call__(self, inp, targets):

        assert len(inp.shape)==4

        if np.random.rand() < self.prob: # probability of applying augmentation
        
            ## distortion with perlin noise and norm
            zres, xres, yres = inp.shape[1:]
            # Initialise Coordinates
            X, Z, Y = np.meshgrid(np.arange(xres), np.arange(zres), np.arange(yres))
    
            # Generate Perturbations with Perlin Noise
            if isinstance(self.grid_points, int):
                GP = [self.grid_points, self.grid_points, self.grid_points]
            else:
                GP = self.grid_points
            RES = [self.next_power_of_two(zres), self.next_power_of_two(xres), self.next_power_of_two(yres)]
            
            noise_z = 2*(self.p*zres)*(self.norm(perlin_optim_3d(RES, GP, sf=self.sf, octaves=1))-.5)
            noise_x = 2*(self.p*xres)*(self.norm(perlin_optim_3d(RES, GP, sf=self.sf, octaves=1))-.5)
            noise_y = 2*(self.p*yres)*(self.norm(perlin_optim_3d(RES, GP, sf=self.sf, octaves=1))-.5)

            # crop to size
            noise_z = noise_z[:zres,:xres,:yres]; noise_x = noise_x[:zres,:xres,:yres]; noise_y = noise_y[:zres,:xres,:yres]
    
            # Update Coordinates
            Z_new = Z + noise_z.astype(int); X_new = X + noise_x.astype(int); Y_new = Y + noise_y.astype(int)
    
            # Apply Distortions
            new_inp = map_coordinates(inp[0].astype(float), [Z_new,X_new,Y_new], order=self.order, mode='constant')[np.newaxis]
            new_inp = new_inp.astype(inp.dtype)
            new_targs = []
            for targ in targets:
                if targ.dtype == bool:
                    new_targ = map_coordinates(targ.astype(float), [Z_new,X_new,Y_new], order=self.order, mode='constant') > .5
                else:
                    new_targ = map_coordinates(targ.astype(float), [Z_new,X_new,Y_new], order=self.order, mode='constant').astype(targ.dtype)
                    
                new_targs.append(new_targ)
                    
            return new_inp, new_targs
        else:
            return inp, targets

    def norm(self, arr):
        return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

    def next_power_of_two(self, x):
        """
        Compute the next power of 2 greater than or equal to x. Use this to 
        make sure the perlin noise generator works for any input shape.
        
        Args:
            x (int): Input value (must be >= 1).
            
        Returns:
            int: The next power of 2 greater than or equal to x.
        """
        if x < 1:
            raise ValueError("Input must be greater than or equal to 1.")
        
        # If x is already a power of 2, return x
        if (x & (x - 1)) == 0:
            return x
    
        # Compute the next power of 2
        return 1 << (x - 1).bit_length()