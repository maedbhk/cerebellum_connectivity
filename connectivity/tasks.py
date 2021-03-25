
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import connectivity.nib_utils as nio
from connectivity.data import Dataset
import connectivity.constants as const

import warnings
warnings.filterwarnings("ignore")


def get_betas(roi, glm, exp, averaging='sess', weighting=True):
    """Get betas for all subjects for `roi`, `glm`, `exp` 
    
    Args: 
        roi (str): 'cerebellum_suit' or 'tesselsWB42'
        glm (str): 'glm7'
        exp (str): 'sc1' or 'sc2'
        averaging (str): 'sess' or 'exp'
        weighting (bool): default is True
    Returns: 
        np array of betas, shape (subjs x tasks x num_voxels; subjs x tasks x num_verts)
    """

    # loop over subjects
    Y_all = []; info_all = pd.DataFrame()
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
        Y, info = data.get_data(averaging=averaging, weighting=weighting)
        
        # append
        Y_all.append(Y)

    # assign new cols to info 
    # don't need to concat across subjs, should be the same
    info['exp'] = exp
    info['roi'] = roi

    # reset index of Y_info (this is the same across subjs)    
    info = info.reset_index()

    # stack the subj data
    Y_stacked = np.stack(Y_all, axis=0)
    
    return Y_stacked, info

def get_betas_summary(rois, exps, glm='glm7', save=True, averaging="none"):
    """calculates summary of subject betas for `rois` and `exps`

    Args: 
        rois (list of str): ['cerebellum_suit', 'tesselsWB162']
        exps (list of str): ['sc1', 'sc2']
        glm (str): default is 'glm7'
        save (bool): default is True

    Returns: 
        summary dataframe of betas averaged across voxels
    """
    dataframe_all = pd.DataFrame()
    for roi in rois:

        for exp in exps:

            betas, info = get_betas(roi=roi, 
                                    glm=glm, 
                                    exp=exp, 
                                    averaging=averaging, 
                                    weighting=True)
            
            # average across voxels
            betas = np.nanmean(betas, axis=2)

            # get number of subjects
            num_subjs, num_tasks = betas.shape

            # add betas to summary
            info_subjs = pd.DataFrame()
            for subj_idx in np.arange(num_subjs):
                info['betas'] = betas[subj_idx,:]
                info['subj'] = subj_idx+1
                info_subjs = pd.concat([info_subjs, info])
            
            dataframe_all = pd.concat([dataframe_all, info_subjs])
            print(f'added betas for {roi} ({exp}) to summary dataframe')
    
    if save:
        pass
    
    return dataframe_all

def plot_task_scatterplot(dataframe, exp='sc1'): 
    """plot scatterplot of beta weights between two rois. 

    Args:   
        dataframe (pd dataframe): dataframe outputf from `get_betas_summary`
        exp (str): 'sc1' or 'sc2'
    """
    for roi in dataframe['roi'].unique():
        if 'cerebellum' in roi:
            roi1 = 'cerebellum'
            roi_x = roi
        else:
            roi2 = 'cortex'
            roi_y = roi

    # get x and y data
    x = dataframe.query(f'roi=="{roi_x}" and exp=="{exp}"').groupby("TN")['betas'].mean().reset_index()
    y = dataframe.query(f'roi=="{roi_y}" and exp=="{exp}"').groupby("TN")['betas'].mean().reset_index()
    df = x.merge(y, on='TN')

    plt.figure(figsize=(10,10))
    sns.regplot(x='betas_x', y='betas_y', data=df, marker="o", color="skyblue")
    plt.xlabel(roi1, fontsize=20)
    plt.ylabel(roi2, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # add annotations one by one with a loop
    for line in range(0,df.shape[0]):
        plt.text(df.betas_x[line]+0.002, df.betas_y[line], df.TN[line], horizontalalignment='left', size='medium', color='black', weight='regular')

    plt.show()

def plot_task_maps_cerebellum(data, data_info, task='Instruct'):
    """plot task maps 
    
    Args: 
        data (np array): shape (17 x 92 x 6937) (subjs x tasks x voxels)
        data_info (pd dataframe): must contain `TN` to index betas
        task (str): default is 'Instruct', inspect data_info to get other task names
    Returns: 
        returns view_all (list of objects), tasks (list of str): task names
    """
    # get group average
    data = np.nanmean(data, axis=0)

    # groupby tasks
    data_info = data_info.groupby('TN').mean().reset_index()

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