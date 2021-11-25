# import libraries
import os
import numpy as np
import nibabel as nib
from numpy.core.fromnumeric import repeat
import pandas as pd
from pathlib import Path
from collections import defaultdict
import glob
from random import seed, sample
import deepdish as dd
from scipy.stats import mode
from SUITPy import flatmap
from SUITPy import atlas as catlas
from nilearn.surface import load_surf_data
from scipy.stats.mstats import gmean

import connectivity.constants as const
import connectivity.io as cio
from connectivity import model
from connectivity import data as cdata
from connectivity import nib_utils as nio
from connectivity.visualize import get_best_models

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
        model_name (str): model_name (folder in conn_train_dir). Has to follow naming convention <method>_<cortex>_alpha_<num>
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

def cortical_surface_voxels(
    model_name, 
    cortex,
    train_exp='sc1',
    weights='nonzero',
    save_maps=True
    ):
    """save surface maps for cerebellum (count number of non-zero cortical coef)

    Args:
        model_name (str): full name of trained model. Has to follow naming convention <method>_<cortex>_alpha_<num>
        train_exp (str): 'sc1' or 'sc2'
        weights (str): 'positive' or 'nonzero' (neg. & pos.). default is 'nonzero'
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = []
    for subj in const.return_subjs:
        model_fnames.append(os.path.join(fpath, f'{model_name}_{subj}.h5'))

    n_models = len(model_fnames)

    cereb_all_count = []; cereb_all_percent = []
    subjs_all = defaultdict(list)
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)

        if weights=='positive':
            data.coef_[data.coef_ <= 0] = np.nan
        elif weights=='nonzero':
            data.coef_[data.coef_ == 0] = np.nan
        
        # count number of non-zero weights
        data_nonzero = np.count_nonzero(~np.isnan(data.coef_,), axis=1)
        n_cereb, n_cortex  = data.coef_.shape

        data_nonzero_percent = np.divide(data_nonzero,  n_cortex)*100
        cereb_all_count.append(data_nonzero)
        cereb_all_percent.append(data_nonzero_percent)

        model_name = Path(model_fname).stem
        method = model_name.split('_')[0]
        subj = model_name.split('_')[-1]
        data = {'method': method, 
                'subj': subj, 
                'train_exp': train_exp,
                'weights': weights,
                'cortex': cortex, 
                'reg_names': 'all-regs',
                'atlas': 'all-voxels'
                }
        for k,v in data.items():
            subjs_all[k].append(v)
    
    roi_avrg = np.nanmean(np.stack(cereb_all_count, axis=0), axis=1)
    subjs_all.update({'count': roi_avrg,
                    'percent': np.divide(roi_avrg, n_cortex)*100
                    })
    
    if save_maps:
        # save maps to disk for cerebellum
        save_maps_cerebellum(data=np.stack(cereb_all_count, axis=0), fpath=os.path.join(fpath, f'group_lasso_count_{weights}_cerebellum'))
        save_maps_cerebellum(data=np.stack(cereb_all_percent, axis=0), fpath=os.path.join(fpath, f'group_lasso_percent_{weights}_cerebellum'))

    return subjs_all

def cortical_surface_rois(
    model_name, 
    cortex,
    alpha,
    train_exp='sc1',
    atlas='MDTB10',
    weights='nonzero'
    ):
    """save weight summary for cerebellar rois (count number of non-zero cortical coef)

    Args:
        model_name (str): full name of trained model. Has to follow naming convention <method>_<cortex>_alpha_<num>
        train_exp (str): 'sc1' or 'sc2'
        weights (str): 'positive' or 'absolute' (neg. & pos.). default is 'positive'
    Returns: 
        data_all (dict): contains keys `count`, `percent`
    """
    dirs = const.Dirs(exp_name=train_exp)

    # full path to best model
    fpath = os.path.join(dirs.conn_train_dir, model_name)
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # get alpha for each model
    method = model_name.split('_')[0] # model_name

    data_all = defaultdict(list)
    for subj in const.return_subjs:
        roi_betas, reg_names, colors = average_region_data(subj,
                                exp=train_exp, cortex=cortex, 
                                atlas=atlas, method=method, alpha=int(alpha), 
                                weights=weights, average_subjs=False)
        # count number of non-zero weights
        data_nonzero = np.count_nonzero(~np.isnan(roi_betas,), axis=1)
        n_cereb, n_cortex = roi_betas.shape

        data = {'count': data_nonzero,
                'percent': np.divide(data_nonzero,  n_cortex)*100,
                'subj': np.repeat(subj, n_cereb),
                'method': np.repeat(method, n_cereb),
                'cortex': np.repeat(cortex, n_cereb),
                'reg_names': reg_names,
                'weights': np.repeat(weights, n_cereb),
                'train_exp': np.repeat(train_exp, n_cereb),
                'atlas': np.repeat(atlas, n_cereb),
                }
        # colors_dict = pd.DataFrame.to_dict(pd.DataFrame(colors, columns=['R','G','B','A']), orient='list')
        # data.update(colors_dict)
        
        for k, v in data.items():
            data_all[k].extend(v)
        
    return data_all

def threshold_data(
    data, 
    threshold=5
    ):
    """threshold data (2d np array) taking top `threshold` % of strongest data

    Args: 
        data (np array): weight matrix; shape n_cerebellar_regs x n_cortical_regs
        threshold (int): default is 5 (takes top 5% of strongest data)

    Returns:
        data (np array); same shape as `data`. NaN replaces all data below threshold
    """
    num_vert = data.shape[1]

    thresh_regs = round(num_vert*(threshold*.01))
    sorted_roi = np.argsort(-data, axis=1)
    sorted_idx = sorted_roi[:,thresh_regs:]
    np.put_along_axis(data, sorted_idx, np.nan, axis=1)

    return data, sorted_idx

def average_region_data(
    subjs,
    exp='sc1',
    cortex='tessels1002',
    atlas='MDTB10',
    method='ridge',
    alpha=8,
    weights='nonzero',
    average_subjs=True
    ):
    """fit model using `method` and `alpha` for cerebellar `atlas` and `cortex`.
    
    Return betas thresholded using `weights`.

    Args: 
        subjs (list of subjs): 
        exp (str): default is 'sc1'. other option: 'sc2'
        cortex (str): cortical atlas. default is 'tessels1002'
        atlas (str): cerebellar atlas. default is 'MDTB10'
        method (str): default is 'ridge'
        alpha (int): default is 8
        weights (str): default is 'nonzero'. other option is 'positive'
        average_subjs (bool): average betas across subjs? default is True
    Returns:    
       roi_mean (shape; n_cerebellar_regs x n_cortical_regs)
       reg_names (shape; n_cerebellar_regs,) 
       colors (shape; n_cerebellar_regs,)
    """
    # set directory
    dirs = const.Dirs(exp_name=exp)

    # fetch `atlas`
    atlas_dir = os.path.join(dirs.cerebellar_atlases, 'king_2019')
    cerebellum_nifti = os.path.join(atlas_dir, f'atl-{atlas}_space-SUIT_dseg.nii')
    cerebellum_gifti = os.path.join(atlas_dir, f'atl-{atlas}_dseg.label.gii')

    if not os.path.isfile(cerebellum_nifti):
        # print(Exception('please download atlases using SUITPy.atlas fetchers'))
        catlas.fetch_king_2019(data='atl', data_dir=dirs.cerebellar_atlases)
        catlas.fetch_buckner_2011(data_dir=dirs.cerebellar_atlases)
        catlas.fetch_diedrichsen_2009(data_dir=dirs.cerebellar_atlases)

    # Load and average region data (average all subjs)
    Ydata = cdata.Dataset(experiment=exp, roi="cerebellum_suit", subj_id=subjs) 
    Ydata.load()
    Xdata = cdata.Dataset(experiment=exp, roi=cortex, subj_id=subjs) # const.return_subjs)
    Xdata.load()
    
    if average_subjs:
        Ydata.average_subj()
        Xdata.average_subj()

    # Read MDTB atlas
    index = cdata.read_suit_nii(cerebellum_nifti)
    Y, _ = Ydata.get_data('sess', True)
    X, _ = Xdata.get_data('sess', True)
    Ym, reg = cdata.average_by_roi(Y,index)

    reg_names =nio.get_gifti_labels(cerebellum_gifti)[1:]
    colors,_,_ = nio.get_gifti_colors(cerebellum_gifti, ignore_0=True)

    # Fit Model to region-averaged data 
    if method=='lasso':
        model_name = 'Lasso'
    elif method=='ridge':
        model_name = 'L2regression'

    fit_roi = getattr(model, model_name)(**{'alpha':  np.exp(alpha)})
    fit_roi.fit(X,Ym)
    roi_mean = fit_roi.coef_[1:]

    if weights=='positive':
        roi_mean[roi_mean <= 0] = np.nan
    elif weights=='nonzero':
        roi_mean[roi_mean == 0] = np.nan
    
    return roi_mean, reg_names, colors

def regions_cortex(
    roi_betas,
    reg_names,
    cortex, 
    threshold=5,
    ):
    """save weights maps for `cortex` for cerebellar `reg_names`

    Weights are optionally thresholded. threshold=100 is equivalent to no threshold.

    Args:
        roi_betas (np array): (shape; n_cerebellar_regs x n_cortical_regs)
        reg_names (list of str): shape (n_cerebellar_regs,)
        cortex (str): e.g. 'tessels1002'
        threshold (int or None): default is 5 (top 5%)
    
    Returns: 
        giis (list of giftis; 'L', and 'R' hem)
    """
    hem_names = ['L', 'R']

    # optionally threshold weights based on `threshold`
    giis_hem = []
    for hem in hem_names:

        labels = get_labels_hemisphere(roi=cortex, hemisphere=hem)
        roi_mean_hem = roi_betas[:,labels]

        # optionally threshold data
        if threshold is not None:
            # optionally threshold weights based on `threshold` (separately for each hem)
            roi_mean_hem, _ = threshold_data(data=roi_mean_hem, threshold=threshold)

        # loop over columns
        giis = []
        for dd in roi_mean_hem:
            gii, _ = cdata.convert_cortex_to_gifti(data=dd, atlas=cortex, column_names=reg_names, data_type='func', hem_names=[hem])
            giis.append(gii[0].darrays[0].data)

        # make gifti
        gii_hem = nio.make_func_gifti_cortex(np.vstack(giis).T, column_names=reg_names, anatomical_struct=hem)
        giis_hem.append(gii_hem)
    
    return giis_hem

def distances_cortex(
    roi_betas,
    reg_names,
    colors,
    cortex, 
    threshold=5,
    metric='gmean'
    ):
    """computes stats for distances (using `metric`) for `cortex` for `reg_names`

    Args:
        roi_betas (np array): (shape; n_cerebellar_regs x n_cortical_regs)
        reg_names (list of str): (shape; n_cerebellar_regs,)
        colors (np array): (shape; n_cerebellar_regs,)
        cortex (str): eg. 'tessels1002'
        threshold (int or None): default is 5 (top 5%)
        metric (str): default is 'gmean'
    
    Returns: 
        data (dict)
    """
    # get data
    num_cols, num_vert = roi_betas.shape

    hem_names = ['L', 'R']

    # optionally threshold weights based on `threshold`
    data = {}; roi_dist_all = []
    for hem in hem_names:

        labels = get_labels_hemisphere(roi=cortex, hemisphere=hem)
        roi_mean_hem = roi_betas[:,labels]

        # optionally threshold data
        if threshold is not None:
            # optionally threshold weights based on `threshold` (separately for each hem)
            roi_mean_hem, _ = threshold_data(data=roi_mean_hem, threshold=threshold)
        
        # distances
        distances_sparse = sparsity_cortex(coef=roi_mean_hem, roi=cortex, metric=metric, hem_names=[hem])[hem]
        roi_dist_all.append(distances_sparse)
        
    # save to disk  
    data.update({'distance': np.hstack(roi_dist_all),
                'hem': np.hstack([np.repeat('L', num_cols), np.repeat('R', num_cols)]),
                'labels': np.tile(reg_names, 2), 
                'threshold': np.repeat(threshold*.01, num_cols*len(hem_names)),
                'metric': np.repeat(metric, num_cols*len(hem_names)),
                'cortex': np.repeat(cortex, num_cols*len(hem_names)),
                })

    df_color = pd.DataFrame(np.vstack([colors, colors]), columns=['R','G', 'B', 'A'])
    data.update(pd.DataFrame.to_dict(df_color, orient='list'))

    return data

def sparsity_cortex(
    coef, 
    roi, 
    metric='gmean', 
    hem_names=['L', 'R']
    ):
    """Compute mean of non-zero cortical distances (measure of cortical sparsity)
    Args: 
        coef (np array): (shape; n_cerebellar_regs (or voxels) x n_cortical_regs)
        roi (str): cortex name e.g., 'tessels1002'
        metric (str): 'gmean', 'nanmean', 'median'
    Returns: 
        data (dict): dict with keys: `L`, `R`
        values are each an np array of shape (voxels x 1)
    """

    # get distances between cortical regions; shape (num_reg x num_reg)
    distances = cdata.get_distance_matrix(roi)[0]

    data = {}
    for hem in hem_names:

        labels = get_labels_hemisphere(roi, hemisphere=hem)

        # index by `hem`
        coef_hem = coef
        if coef.shape[1]==distances.shape[0]:
            coef_hem = coef[:, labels]
        
        dist_hem = distances[labels,:][:,labels]

        # get shape of coefficients
        regs, _ = coef_hem.shape

        # loop over voxels
        nonzero_dist = np.zeros((regs, ))
        for reg in np.arange(regs):

            coef_hem = np.nan_to_num(coef_hem)
            labels_arr = np.nonzero(coef_hem[reg,:])[0]

            # pairwise distances for nonzero `labels`
            dist_mat = dist_hem[labels_arr,:][:,labels_arr]
            dist_labels = dist_mat[np.triu_indices_from(dist_mat, k=1)]

            if metric=='gmean':
                nonzero_dist[reg] = gmean(dist_labels)
            elif metric=='nanmean':
                nonzero_dist[reg] = np.nanmean(dist_labels)
            elif metric=='nanmedian':
                nonzero_dist[reg] = np.nanmedian(dist_labels)

        # add to dict
        data.update({hem: nonzero_dist})

    if len(hem_names)>1:
        # get average across hemispheres
        data.update({'L_R': np.nanmean([data['L'], data['R']], axis=0)})

    return data

def dispersion_cortex(
    roi_betas,
    reg_names,
    colors,
    cortex
    ):
    """Caluclate spherical dispersion for the connectivity weights 

    Args:
        roi_betas (np array):
        reg_names (list of str):
        colors (np array):
        cortex (str):
    
    Returns: 
        dataframe (pd dataframe)
    """
    # get data
    num_roi, num_parcel = roi_betas.shape

    hem_names = ['L', 'R']

    # optionally threshold weights based on `threshold`
    data = {}; roi_dist_all = []
    dist,coord = cdata.get_distance_matrix(cortex)

    df = pd.DataFrame()

    for h,hem in enumerate(hem_names):

        labels = get_labels_hemisphere(roi=cortex, hemisphere=hem)
        # weights 
        
        # Calculate spherical STD as measure 
        # Get coordinates and move back to 0,0,0 center
        coord_hem = coord[labels,:]
        coord_hem[:,0]=coord_hem[:,0]-(h*2-1)*500        

        # Now compute a weoghted spherical mean, variance, and STD 
        # For each tessel, the weigth w_i is the connectivity weights with negative weights set to zero
        # also set the sum of weights to 1 
        w = roi_betas[:,labels]
        w[w<0]=0
        w = w / w.sum(axis=1).reshape(-1,1)
    
        # We then define a unit vector for each tessel, v_i: 
        v = coord_hem.copy().T 
        v=v / np.sqrt(np.sum(v**2,axis=0))

        # Weighted average vector =sum(w_i*v_i)
        # R is the length of this average vector 

        R = np.zeros((num_roi,))
        for i in range(num_roi):

            mean_v = np.sum(w[i,:] * v,axis=1)
            R[i] = np.sqrt(np.sum(mean_v**2))

            #Check with plot
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(w[i,:]*v[0,:],w[i,:]*v[1,:],w[i,:]*v[2,:])
            # ax.scatter(mean_v[0],mean_v[1],mean_v[2])
            # pass

        V = 1-R # This is the Spherical variance
        Std = np.sqrt(-2*np.log(R)) # This is the spherical standard deviation
        df1 = pd.DataFrame({'Variance':V,'Std':Std,'hem':h*np.ones((num_roi,)),'roi':np.arange(num_roi)})
        df = pd.concat([df,df1])
    return df

def get_labels_hemisphere(
    roi, 
    hemisphere
    ):
    """Get labels for `roi` for `hemisphere`
    
    Args: 
        roi (str): example is 'tessels1002'
        hemisphere (str): 'L' or 'R'
    Returns: 
        1D np array of labels
    """
    dirs = const.Dirs(exp_name='sc1')
    
    gii_path = os.path.join(dirs.reg_dir, 'data', 'group', f'{roi}.{hemisphere}.label.gii')
    labels = load_surf_data(gii_path)

    # get min, max labels for each hem
    min_label = np.nanmin(labels[labels!=0])
    max_label = np.nanmax(labels[labels!=0])

    # get labels per hemisphere
    labels_hem = np.arange(min_label-1, max_label)
    
    return labels_hem

def best_weights(
    train_exp='sc1',
    method='ridge',
    ):
    """Get group average model weights for best trained model

    Args: 
        train_exp (str): default is 'sc1'
        method (str): default is 'L2regression'
    Returns: 
        group_weights (n-dim np array)

    """
    # get best models
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
    
