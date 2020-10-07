import numpy as np
import nibabel as nib
from nilearn import image
from scipy import stats

from nilearn._utils import check_niimg
from nilearn._utils.niimg import _get_data, _safe_get_data
from nilearn._utils.niimg_conversions import _check_same_fov

from connectivity.constants import Defaults, Dirs

"""
Created on Mon Aug  3 17:40:09 2020
includes functions for visualizations
​
@authors: Maedbh King and Ladan Shahshahani
"""

def convert_to_vol(Y, vox_indx, mask):
    """
    This function converts 1D numpy array data to 3D vol space, and returns nib obj
    that can then be saved out as a nifti file
    Args:
        Y (numpy array): voxel data, shape (num_vox, )
        vox_indx (numpy array of int): non-zero indices for grey matter voxels in cerebellum, shape (num_vox, )
        mask (nib obj): nib obj of mask
    
    Returns: 
        Nib Obj
    
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

    return make_nifti_obj(vol_data=vol_data, affine_mat=mat)

def make_nifti_obj(vol_data, affine_mat):
    """ makes nifti obj 
        Args: 
            vol_data (numpy array): data in vol space (xyz)
            affine_mat (numpy array): affine transformation matrix
        Returns: 
            Nib Obj
    """
    return nib.Nifti1Image(vol_data, affine_mat)

def save_nifti_obj(nib_obj, fpath):
    """ saves nib obj to nifti file
        Args: 
            nib_obj (Niimg-like object): contains vol data in nib obj
            fpath (str): full path to nib_obj
        Returns: 
            saves nifti file to fpath
    """

    nib.save(nib_obj, fpath)
    print(f'saved {fpath}')

def calc_nifti_average(imgs, fpath):
    """ calculates average nifti image
        Args: 
            imgs (list): iterable of Niimg-like objects
            out_path (str): full path to averaged image
        Returns: 
            saves averaged nifti file to disk
    """
    mean_img = image.mean_img(imgs) # get average of images from `filenames` list

    # save nifti to disk
    save_nifti_obj(nib_obj=mean_img, fpath=fpath)

def calc_nifti_mode(imgs, fpath):
    # mode_img = image.math_img("np.argmax(np.bincount(img))", img=imgs)
    # image.math_img doesn't calculate the mode because np doesn't have a mode function ...
    # therefore, I borrowed the code from image.math_img and input scipy.stats

    # Check that input images are valid niimg and have a compatible shape
    # and affine.
    niimgs = []
    for image in imgs: # imgs.values():
        niimgs.append(check_niimg(image))
    _check_same_fov(*niimgs, raise_error=True)

    # Computing input data as a dictionary of numpy arrays. Keep a reference
    # niimg for building the result as a new niimg.
    niimg = None
    data_dict = {}
    # for key, img in imgs.items():
    for key, img in enumerate(imgs):
        niimg = check_niimg(img)
        data_dict[key] = _safe_get_data(niimg)

    data_dict['stats'] = stats
    formula = "stats.mode(img)"
    img = imgs

    result = eval(formula, data_dict)

    # get mode img
    mode_img = new_img_like(niimg, result, niimg.affine)

    keyboard

    # save nifti to disk
    save_nifti_obj(nib_obj=mode_img, fpath=fpath)


