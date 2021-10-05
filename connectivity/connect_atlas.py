import os
import numpy as np
import glob
import nibabel as nib
from scipy.stats import mode
from nilearn.surface import load_surf_data
from SUITPy import flatmap

import connectivity.constants as const
import connectivity.io as cio
import connectivity.model as model
from connectivity import data as cdata
from connectivity import nib_utils as nio

def _get_data(subj, exp, glm, atlas):
    """get cortical and cerebellar data 

    Args: 
        subj (str): 's01' for example
        exp (str): 'sc1' or 'sc2'
        glm (str): 'glm7'
        atlas (str): 'yeo7' etc.

    Returns: 
        Y, Y_info, X, X_info
    """
    # Get the data
    Ydata = cdata.Dataset(
        experiment=exp,
        glm=glm,
        subj_id=subj,
        roi='cerebellum_suit',
    )

    # load mat
    Ydata.load_mat()
    Y, Y_info = Ydata.get_data(averaging='sess', weighting=True)

        # Get the data
    Xdata = cdata.Dataset(
        experiment=exp,
        glm=glm,
        subj_id=subj,
        roi=atlas,
    )

    # load mat
    Xdata.load_mat()
    X, X_info = Xdata.get_data(averaging='sess', weighting=True)

    return Y, Y_info, X, X_info

def model_wta(subjs, exp, glm, atlas, crossvalidate=False):

    labels_all = []
    for subj in subjs:

        Y, Y_info, X, X_info = _get_data(subj, exp, glm, atlas)

        new_model = getattr(model, 'WTA')(**{"positive": True})

        # cross the sessions
        if crossvalidate:
            Y = np.r_[Y[Y_info.sess == 2, :], Y[Y_info.sess == 1, :]]
        
        # fit model
        new_model.fit(X, Y)
        labels_all.append(new_model.labels)

    return np.vstack(labels_all)

def dice_coefficient(atlas1, atlas2, mask=None):
    """Calculate dice coefficient between two atlases

    ***scipy.spatial.distance.pdist calculates dice coefficient but doesn't deal with NaN values****

    Args: 
        atlas1 (str or nib obj):
        atlas2 (str or nib obj):
        mask (None or str): if mask is provided, extracts data from mask (speeds up computation)

    Returns: 
        Dice (np array): shape (num_labels x num_labels)
    """
    if isinstance(atlas1, str):
        atlas1 = nib.load(atlas1)
    
    if isinstance(atlas2, str):
        atlas2 = nib.load(atlas2)

    # get data
    atlas1 = atlas1.get_data()
    atlas2 = atlas2.get_data()

    # extract data from mask for `atlas1` and `atlas2` if provided
    if mask is not None:
        atlas1 = nio.mask_vol(mask, vol=atlas1, output='2D')
        atlas2 = nio.mask_vol(mask, vol=atlas2, output='2D')

    # get labels
    labels1 = np.unique(atlas1)
    labels1 = labels1[labels1!=0]
    labels2 = np.unique(atlas2)
    labels2 = labels2[labels2!=0]

    Dice = np.zeros((labels1.size, labels2.size))
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            dice = np.sum(atlas1[atlas2==label2]==label1)*2.0 / (np.sum(atlas1[atlas1==label1]==label1) + np.sum(atlas2[atlas2==label2]==label2))
            
            Dice[int(i)][int(j)]=float(dice)

            if dice > 1 or dice < 0:
                raise ValueError(f"Dice coefficient is greater than 1 or less than 0 ({dice}) at atlas1: {label1}, atlas2: {label2}")

    return Dice

def save_maps_cerebellum(
    data, 
    fpath='/',
    group='nanmean', 
    gifti=True, 
    nifti=False, 
    column_names=[], 
    label_RGBA=[],
    label_names=[],
    ):
    """Takes data (np array), averages along first dimension
    saves nifti and gifti map to disk

    Args: 
        data (np array): np array of shape (N x 6937)
        fpath (str): save path for output file
        group (bool): default is 'nanmean' (for func data), other option is 'mode' (for label data) 
        gifti (bool): default is True, saves gifti to fpath
        nifti (bool): default is False, saves nifti to fpath
        column_names (list):
        label_RGBA (list):
        label_names (list):
    Returns: 
        saves nifti and/or gifti image to disk, returns gifti
    """
    num_cols, num_vox = data.shape

    # get mean or mode of data along first dim (first dim is usually subjects)
    if group=='nanmean':
        data = np.nanmean(data, axis=0)
    elif group=='mode':
        data = mode(data, axis=0)
        data = data.mode[0]
    else:
        print('need to group data by passing "nanmean" or "mode"')

    # convert averaged cerebellum data array to nifti
    nib_obj = cdata.convert_cerebellum_to_nifti(data=data)[0]
    
    # save nifti(s) to disk
    if nifti:
        nib.save(nib_obj, fpath + '.nii')

    # map volume to surface
    surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT", stats=group)

    # make and save gifti image
    if group=='nanmean':
        gii_img = flatmap.make_func_gifti(data=surf_data, column_names=column_names)
        out_name = 'func'
    elif group=='mode':
        gii_img = flatmap.make_label_gifti(data=surf_data, label_names=label_names, column_names=column_names, label_RGBA=label_RGBA)
        out_name = 'label'
    if gifti:
        nib.save(gii_img, fpath + f'.{out_name}.gii')
    
    return gii_img