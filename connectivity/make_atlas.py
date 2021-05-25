import os
import numpy as np
import glob
import nibabel as nib
from scipy.stats import mode
from nilearn.surface import load_surf_data
import SUITPy.flatmap as flatmap

import connectivity.constants as const
import connectivity.io as cio
import connectivity.model as model
from connectivity import data as cdata

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

def model_wta(exp, glm, atlas, crossvalidate=False):

    labels_all = []
    # for subj in const.return_subjs:
    for subj in ['s02', 's03']:

        Y, Y_info, X, X_info = _get_data(subj, exp, glm, atlas)

        new_model = getattr(model, 'WTA')(**{"positive": True})

        # cross the sessions
        if crossvalidate:
            Y = np.r_[Y[Y_info.sess == 2, :], Y[Y_info.sess == 1, :]]
        
        # fit model
        new_model.fit(X, Y)
        labels_all.append(new_model.labels)

    return np.vstack(labels_all)

def get_label_colors(atlas, hem='L'):
    """get rgba for `atlas`

    Args: 
        atlas (str): 'yeo7' etc.
        hem (str): 'L' or 'R' (colors should be the same regardless)
    Returns: 
        rgba (np array): shape num_labels x num_rgba
    """
    dirs = const.Dirs()

    if atlas=='yeo7':
        atlas = 'Yeo_JNeurophysiol11_7Networks'
    elif atlas=='yeo17':
        atlas = 'Yeo_JNeurophysiol11_17Networks'

    img = nib.load(os.path.join(dirs.fs_lr_dir, f'{atlas}.32k.{hem}.label.gii'))
    labels = img.labeltable.labels

    rgba = np.zeros((len(labels),4))
    for i,label in enumerate(labels):
        rgba[i,] = labels[i].rgba

    return rgba

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