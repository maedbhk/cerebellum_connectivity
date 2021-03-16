# import packages
from pathlib import Path
import nibabel as nib
from nilearn.image import mean_img
import os

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

