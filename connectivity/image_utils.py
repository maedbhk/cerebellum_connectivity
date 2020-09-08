import numpy as np
import nibabel as nib

from connectivity.constants import Defaults, Dirs

"""
Created on Mon Aug  3 17:40:09 2020
includes functions for visualizations
​
@authors: Ladan Shahshahani and Maedbh King
"""

def make_nifti_obj(Y, vox_indx, mask):
    """
    This function takes numpy arrays and creates nifti files for cerebellum data
    Then, vol2surf or map2surf can be used to transfer the volume space to flatmap
    Args:
        Y (numpy array): voxel data, shape (num_vox, )
        vox_indx (numpy array of int): non-zero indices for grey matter voxels in cerebellum,
        mask (nib obj): nib obj of mask
    
    Returns: 
        Nib obj
    
    """
    # get dat, mat, and dim from the mask
    dat = mask.get_fdata()
    dim = dat.shape
    mat = mask.affine
    
    # initialise xyz voxel data
    vol_data = np.zeros((dim[0], dim[1], dim[2], 1))
    
    # get the xyz of the nonZeroVoxels used in estimating connectivity weights
    (x, y, z)= np.unravel_index(vox_indx, mask.shape, 'F')
    
    num_vox = len(Y)
    for i in range(num_vox):
        vol_data[x[i], y[i], z[i]] = Y[i]

    nib_obj = nib.Nifti1Image(vol_data, mat)
        
    return nib_obj

def save_to_nifti(nib_obj, fpath):
    """ saves nib obj to nifti file
        Args: 
            nib_obj: contains vol data in nib obj
        Returns: 
            saves nifti file to fpath
    """

    nib.save(nib_obj, fpath)