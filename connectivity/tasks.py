
import numpy as np

import connectivity.nib_utils as nio
from connectivity.data import Dataset
import connectivity.constants as const


def get_betas(roi, glm, exp, averaging='sess', weighting=True, group_avrg=True):
    """Get betas for all subjects (or average across subjs) for `roi`, `glm`, `exp` 
    
    Args: 
        roi (str): 'cerebellum_suit' or 'tessels042'
        glm (str): 'glm7'
        exp (str): 'sc1' or 'sc2'
        averaging (str): 'sess' or 'exp'
        weighting (bool): default is True
        group_avrg (bool): default is True
    Returns: 
        np array of betas, shape (tasks x num_voxels; tasks x num_verts)
        or (subjs x tasks x num_voxels; subjs x tasks x num_verts)
    """

    # loop over subjects
    Y_all = []
    for subj in const.return_subjs:
        
        # Get the data
        data = Dataset(
            experiment=exp,
            glm=glm,
            subj_id=subj,
            roi=roi,
        )

        # load mat
        data.load_mat()

        # load data
        Y, Y_info = data.get_data(averaging=averaging, weighting=weighting)

        Y_all.append(Y)

    # reset index of Y_info (this is the same across subjs)    
    Y_info = Y_info.reset_index()

    # stack the subj data
    Y_stacked = np.stack(Y_all, axis=0)
    
    if group_avrg:
        return np.nanmean(Y_stacked, axis=0), Y_info
    else:
        return Y_stacked, Y_info
    

def plot_task_maps_cerebellum(data, data_info, task='Instruct'):
    """plot task maps 
    
    Args: 
        data (np array): shape (92 x 6937)
        data_info (pd dataframe): must contain `TN` to index betas
        task (str): default is 'Instruct', inspect data_info to get other task names
    Returns: 
        returns view_all (list of objects), tasks (list of str): task names
    """
    # get betas for `task`
    idx = data_info[data_info['TN']==task].index

    # average 
    betas = np.nanmean(data[idx,:], axis=0)
    betas = betas.reshape(1, len(betas))

    # convert betas to gifti 
    gii_img = nio.save_maps_cerebellum(data=betas,
                                    column_names=task,
                                    gifti=False,
                                    nifti=False,
                                    group_average=False)

    # plot cerebellum
    view = nio.view_cerebellum(data=gii_img.darrays[0].data)
    
    return view