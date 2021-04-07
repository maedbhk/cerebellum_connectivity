import os
import numpy as np
from collections import defaultdict
from nilearn.surface import load_surf_data

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
    dist_sum_all = []; dist_var_all = []
    dist_mean_all = []
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

        # get std of distances for vox
        dist_mean = np.nanmean(np.nanmean(dist_vox,axis=0))

        dist_sum_all.append(dist_sum)
        dist_var_all.append(dist_var)
        dist_mean_all.append(dist_mean)

    return {'mean_distances_vox': dist_mean_all,
            'sum_var_distances_vox':  list(np.array(dist_sum_all) / np.array(dist_var_all))}