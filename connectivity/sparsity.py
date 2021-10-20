import os
import numpy as np
from nilearn.surface import load_surf_data
from scipy.stats.mstats import gmean

from connectivity import data as cdata
from connectivity import constants as const
        
def get_labels_hemisphere(roi, hemisphere):
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

def threshold_weights(weights, threshold=0.1):
    """Threshold weights
    
    Returns sorted indices from largest-smallest
    
    Args: 
        weights (2d np array): weights of shape (num_voxels x num_roi)
        threshold (int): default is 0.1
    Returns: 
        weights_indices_thresh (2d np array) sorted weight indices of shape (num_voxels x num_roi)
    """

    # sort data (largest-smallest weight)
    weights_hem_indx = (-weights).argsort(axis=1)
    
    # get indices for top `threshold` of cortical tessels
    threshold_idx = int(np.round(weights_hem_indx.shape[1] * threshold))

    # get data indexed by top `threshold`
    weight_indices_thresh = weights_hem_indx[:, :threshold_idx]
    
    return weight_indices_thresh

def weight_distances(weights, distances):
    tmp = np.nanmean((distances @ weights.T), axis=0)
    tmp[tmp==0]=np.nan
    return {'weighted_distances_vox': list(tmp)}

def get_distance_weights(weight_indices, distances):
    """Calculate the sum/var/std of distances for weight_indices
    
    Args: 
        weight_indices (2d np array): shape (num_voxels x num_roi)
        distances (2d np array): shape (num_roi x num_roi)
    Returns: 
        data_dict (dict) contains np array of sum/var/std of distances
    """
    # get num voxels
    num_vox = weight_indices.shape[0]

    # loop over voxels
    dist_sum_var_all = []
    for vox in np.arange(num_vox):

        # get top indices for vox
        data_vox = weight_indices[vox,:]

        # get array of distances for vox
        dist_vox = distances[data_vox][:,data_vox]
        dist_vox[dist_vox==0]=np.nan

        # get sum of distances for vox
        dist_sum = np.nansum(np.nansum(dist_vox, axis=0))

        # get variances of distances for vox
        dist_var = np.nanvar(np.nanvar(dist_vox, axis=0))

        dist_sum_var_all.append((dist_sum / dist_var))
    
    # zeros should be nan
    dist_sum_var_all[dist_sum_var_all==0]=np.nan

    return {'sum_var_distances_vox':  dist_sum_var_all}

def calc_distances(
    coef, 
    roi, 
    metric='gmean', 
    hem_names=['L', 'R']
    ):
    """Compute mean of cortical distances
    Args: 
        coef (np array): (shape; n_cerebellar_regs (or voxels) x n_cortical_regs)
        roi (str): cortex name e.g., 'tessels1002'
        metric (str): 'gmean', 'nanmean', 'median'
    Returns: 
        data (dict): dict with keys: left hemi, right hemi,
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