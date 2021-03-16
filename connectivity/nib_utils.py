# import packages
from pathlib import Path
import nibabel as nib
import numpy as np
from nilearn.image import mean_img
import os
import SUITPy.flatmap as flatmap

import connectivity.nib_utils as nio
from connectivity.data import convert_cerebellum_to_nifti

def nib_load(fpath):
    """ load nifti from disk

    Args: 
        fpath (str): full path to nifti img
    Returns: 
        returns nib obj
    """
    return nib.load(fpath)

def nib_save(img, fpath):
    """Save nifti to disk
    
    Args: 
        img (nib obj): 
        fpath (str): full path to nifti
    Returns: 
        Saves img to disk
    """
    nib.save(img, fpath)

def nib_mean(imgs):
    """ Get mean of nifti objs

    Args: 
        imgs (list): list of nib objs
    Returns: 
        mean nib obj
    """
    return mean_img(imgs)

def save_maps_cerebellum(data, fpath, fname, group_average=True, save_nifti=True):
    """Takes list of np arrays, averages list and
    saves nifti and gifti map (model predictions) to disk

    Args: 
        data (list): list of np arrays of shape (1 x 6937)
        fpath (str): path where averaged map (nifti and gifti) will be saved
        fname (str): name of nifti/gifti file
        group_average (bool): default is True, averages data np arrays 
        save_nifti (bool): default is True, saves nifti to disk
    Returns: 
        saves nifti and gifti images to disk
    """
    # stack the list of data arrays into nd array
    data = np.stack(data, axis=0)

    # average data
    if group_average:
        data = np.nanmean(data, axis=0)

    # convert averaged cerebellum data array to nifti
    nib_obj = convert_cerebellum_to_nifti(data=data)[0]
    
    # save nifti to disk
    if save_nifti:
        # save nifti to file
        nib_fpath = os.path.join(fpath, f"{fname}.nii")
        nio.nib_save(img=nib_obj, fpath=nib_fpath) # this is temporary (to test bug in map)

    # map volume to surface
    surf_data = flatmap.vol_to_surf([nib_fpath], space="SUIT")

    # make and save gifti image
    gii_img = flatmap.make_func_gifti(data=surf_data, column_names=[fname])
    nio.nib_save(img=gii_img, fpath=os.path.join(fpath, f"{fname}.func.gii"))

def save_maps_cortex():
    pass

