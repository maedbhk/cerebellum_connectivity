# import libraries
import os
import numpy as np
import nibabel as nib
import glob
import deepdish as dd
from scipy.stats import mode
from SUITPy import flatmap
from SUITPy import atlas as dataset

import connectivity.constants as const
import connectivity.io as cio
from connectivity import model
from connectivity import data as cdata
from connectivity import nib_utils as nio
from connectivity.visualize import get_best_model, get_best_models

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
        train_exp,
        save=True
        ):
    """Get weights for trained models. 

    Optionally save out weight maps for cortex and cerebellum separately

    Args: 
        model_name (str): model_name (folder in conn_train_dir)
        cortex (str): cortex model name (example: tesselsWB162)
        train_exp (str): 'sc1' or 'sc2'
    Returns: 
        weights (n-dim np array); saves out cortex and cerebellar maps if `save` is True
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    cereb_weights_all = []; cortex_weights_all = []; weights_all = []
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)
        
        # append cerebellar and cortical weights
        cereb_weights_all.append(np.nanmean(data.coef_, axis=1))
        cortex_weights_all.append(np.nanmean(data.coef_, axis=0))
        weights_all.append(data.coef_)
    
    # stack the weights
    weights_all = np.stack(weights_all, axis=0)

    # save cortex and cerebellum weight maps to disk
    if save:
        save_maps_cerebellum(data=np.stack(cereb_weights_all, axis=0), fpath=os.path.join(fpath, 'group_weights_cerebellum'))

        cortex_weights_all = np.stack(cortex_weights_all, axis=0)
        func_giis, hem_names = cdata.convert_cortex_to_gifti(data=np.nanmean(cortex_weights_all, axis=0), atlas=cortex)
        for (func_gii, hem) in zip(func_giis, hem_names):
            nib.save(func_gii, os.path.join(fpath, f'group_weights_cortex.{hem}.func.gii'))
        print('saving cortical and cerebellar weights to disk')
    
    return weights_all

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
    train_exp,
    cortex='tessels1002',
    atlas='MDTB10',
    weights='positive',
    data_type='func',
    alpha=-2,
    label_names=None,
    label_RGBA=None,
    column_names=None
    ):
    """save lasso maps for cortex

    Args:
        train_exp (str): 'sc1' or 'sc2'
        cortex (str): default is 'tessels1002'
        atlas (str): default is 'MDTB10'
        weights (str): 'positive' or 'absolute' (neg + pos). default is positive
        data_type (str): 'func' or 'label' or 'prob'. default is 'label'
        probabilistic (bool): default is False. 
        label_names (list or None):
        label_RGBA (list or None):
        column_names (list or None):
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # fetch `atlas`
    fpath = dataset.fetch_king_2019(data='atl', data_dir=dirs.cerebellar_atlases)['data_dir']
    cerebellum_nifti = os.path.join(fpath, f'atl-{atlas}_space-SUIT_dseg.nii')
    cerebellum_gifti = os.path.join(fpath, f'atl-{atlas}_dseg.label.gii')

    # Load and average region data (for all subjs)
    Ydata = cdata.Dataset(experiment=train_exp, roi="cerebellum_suit", subj_id=const.return_subjs)
    Ydata.load()
    Ydata.average_subj()
    Xdata = cdata.Dataset(experiment=train_exp, roi=cortex, subj_id=const.return_subjs)
    Xdata.load()
    Xdata.average_subj()

    # Read MDTB atlas
    index = cdata.read_suit_nii(cerebellum_nifti)
    Y, _ = Ydata.get_data('sess', True)
    X, _ = Xdata.get_data('sess', True)
    Ym, reg = cdata.average_by_roi(Y,index)
    reg_names = [f'Region{idx}' for idx in reg[1:]]

    # Fit Lasso Model to region-averaged data 
    fit_roi= model.LASSO(alpha=np.exp(alpha))
    fit_roi.fit(X,Ym)
    roi_mean = fit_roi.coef_

    if weights=='positive':
        roi_mean[roi_mean <= 0] = np.nan
    elif weights=='absolute':
        roi_mean[roi_mean == 0] = np.nan

    # get data (exclude label 0)
    data = roi_mean[1:,:]
    _, num_vert = data.shape

    # functional or label maps
    if data_type=='label':
        labels = np.zeros(num_vert)
        labels[:] = np.nan
        # this is the best solution but it is still hacky (for loops etc.)
        for vert in np.arange(num_vert):
            if not np.all(np.isnan(data[vert,:])):
                labels[vert] = np.nanargmax(data[vert,:]) + 1
        label_names = reg_names
        data = labels
    elif data_type=='func':
        column_names = reg_names
    
    giis, hem_names = cdata.convert_cortex_to_gifti(
                    data=data, 
                    atlas=cortex, 
                    column_names=column_names, 
                    label_names=label_names, 
                    label_RGBA=nio.get_gifti_colors(fpath=cerebellum_gifti),
                    data_type=data_type)

    return giis, hem_names

def best_weights(
    train_exp='sc1',
    method='L2regression',
    ):
    """Get group average model weights for best trained model

    Args: 
        train_exp (str): default is 'sc1'
        method (str): default is 'L2regression'
    Returns: 
        group_weights (n-dim np array)

    """
    # get best L2regression model
    # best_model, cortex = get_best_model(dataframe=None, train_exp=train_exp, method=method)
    models, cortex_names = get_best_models(train_exp='sc1', method=method)

    for (best_model, cortex) in zip(models, cortex_names):

        # get group weights for best model
        weights = weight_maps(model_name=best_model, 
                            cortex=cortex, 
                            train_exp=train_exp, 
                            save=False
                            )
        
        # get group average weights
        group_weights = np.nanmean(weights, axis=0)

        # save best weights to disk
        dirs = const.Dirs(exp_name=train_exp)
        outdir = os.path.join(dirs.conn_train_dir, 'best_weights')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        dd.io.save(os.path.join(outdir, f'{best_model}.h5'), {'weights': group_weights})
    
