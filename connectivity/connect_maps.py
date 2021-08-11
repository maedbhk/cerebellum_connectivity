# import libraries
import os
import numpy as np
import nibabel as nib
import glob
from scipy.stats import mode
import SUITPy.flatmap as flatmap

import connectivity.constants as const
import connectivity.io as cio
from connectivity import data as cdata

def save_maps_cerebellum(
    data, 
    fpath='/',
    group='nanmean', 
    gifti=True, 
    nifti=True, 
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

def weight_maps(
        model_name, 
        cortex, 
        train_exp
        ):
    """Save weight maps to disk for cortex and cerebellum

    Args: 
        model_name (str): model_name (folder in conn_train_dir)
        cortex (str): cortex model name (example: tesselsWB162)
        train_exp (str): 'sc1' or 'sc2'
    Returns: 
        saves nifti/gifti to disk
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    cereb_weights_all = []; cortex_weights_all = []
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)
        
        # append cerebellar and cortical weights
        cereb_weights_all.append(np.nanmean(data.coef_, axis=1))
        cortex_weights_all.append(np.nanmean(data.coef_, axis=0))

    # save maps to disk for cerebellum and cortex
    save_maps_cerebellum(data=np.stack(cereb_weights_all, axis=0), 
                    fpath=os.path.join(fpath, 'group_weights_cerebellum'))

    # save maps to disk for cortex
    data = np.stack(cortex_weights_all, axis=0)
    func_giis, hem_names = cdata.convert_cortex_to_gifti(data=np.nanmean(data, axis=0), atlas=cortex)
    for (func_gii, hem) in zip(func_giis, hem_names):
        nib.save(func_gii, os.path.join(fpath, f'group_weights_cortex.{hem}.func.gii'))

    print('saving cortical and cerebellar weights to disk')

def lasso_maps_cerebellum(
    model_name, 
    train_exp,
    weights='positive'
    ):
    """save lasso maps for cerebellum (count number of non-zero cortical coef)

    Args:
        model_name (str): full name of trained model
        train_exp (str): 'sc1' or 'sc2'
        weights (str): 'positive' or 'absolute' (neg. & pos.). default is 'positive'
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    for stat in ['count', 'percent']:
        cereb_lasso_all = []
        for model_fname in model_fnames:

            # read model data
            data = cio.read_hdf5(model_fname)

            if weights=='positive':
                data.coef_[data.coef_ <= 0] = np.nan
            elif weights=='absolute':
                data.coef_[data.coef_ == 0] = np.nan
            
            # count number of non-zero weights
            data_nonzero = np.count_nonzero(~np.isnan(data.coef_,), axis=1)

            if stat=='count':
                pass # do nothing
            elif stat=='percent':
                num_regs = data.coef_.shape[1]
                data_nonzero = np.divide(data_nonzero,  num_regs)*100
            cereb_lasso_all.append(data_nonzero)

        # save maps to disk for cerebellum
        save_maps_cerebellum(data=np.stack(cereb_lasso_all, axis=0), 
                        fpath=os.path.join(fpath, f'group_lasso_{stat}_{weights}_cerebellum'))

def lasso_maps_cortex(
    model_name, 
    train_exp,
    cortex,
    cerebellum_fpath,
    weights='positive',
    data_type='func'
    ):
    """save lasso maps for cerebellum (count number of non-zero cortical coef)

    There are two different types of maps (given by `map_type`).
    Functional maps are multiple giftis (cortical weights for each cerebellar subregion)
    Label map is one winner-take-all map (each cortical region is tagged with "winning" cerebellar region)

    Args:
        model_name (str): full name of trained model
        train_exp (str): 'sc1' or 'sc2'
        cortex (str):
        cerebellum_fpath (str): full path to cerebellum atlas (*.nii)
        weights (str): 'positive' or 'absolute' (neg + pos). default is positive
        data_type (str): 'func' or 'label'. default is 'label'
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    cortex_all = []
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)
         
        # reshape coefficients
        coef = np.reshape(data.coef_, (data.coef_.shape[1], data.coef_.shape[0]))
        
        # get atlas parcels
        region_number_suit = cdata.read_suit_nii(cerebellum_fpath)

        if weights=='positive':
            coef[coef <= 0] = np.nan
        elif weights=='absolute':
            coef[coef == 0] = np.nan

        # get average for each parcel
        data_mean_roi, region_numbers = cdata.average_by_roi(data=coef, region_number_suit=region_number_suit)

        reg_names = [f'Region{idx}' for idx in region_numbers[1:]]

        # functional or label maps
        if data_type=='func':
            data = data_mean_roi[:,1:]
            column_names = reg_names
            label_names = None
        elif data_type=='label':
            data = np.argmax(np.nan_to_num(data_mean_roi[:,1:]), axis=1) + 1
            column_names = None
            label_names = reg_names

        cortex_all.append(data)

    # save maps to disk for cortex
    group_cortex = np.nanmean(np.stack(cortex_all), axis=0)
    giis, hem_names = cdata.convert_cortex_to_gifti(data=group_cortex, atlas=cortex, column_names=column_names, label_names=label_names, data_type=data_type)

    return giis, hem_names
