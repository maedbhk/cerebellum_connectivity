# import libraries
import pandas as pd
import os
from matplotlib import cm
import nibabel as nib
from SUITPy import flatmap
import matplotlib.pyplot as plt
import numpy as np

import connectivity.constants as const
import connectivity.nib_utils as nio
from connectivity import io 
from connectivity import visualize as summary

def eval_map_names(cortex,exp = 'sc2', methods=['ridge','WTA','lasso']): 
    df = summary.get_summary('eval',exps=exp,summary_name='weighted_all',
                method=methods)
    a = df.name.str.contains(cortex)
    names = np.unique(df.name[a])
    return names

def run_binarize(
    glm='glm7', 
    metric='R', 
    methods=['ridge', 'WTA']
    ):
    """
    Args: 
        glm (str): 'glm7'
        metric (str): evaluation metric: 'R' or 'R2'. default is 'R'
        methods (list of str): default is ['WTA', 'ridge']
    Returns: 
        saves nifti and gifti for difference map between `methods` for evaluated models
    """
    # loop over experiments
    for exp in range(2):
        
        df = summary.eval_summary(exps=[f"sc{exp+1}"])
        df = df[['eval_name', 'eval_X_data']].drop_duplicates() # get unique model names

        # get outpath
        dirs = const.Dirs(exp_name=f"sc{exp+1}", glm=glm)
        fpath = os.path.join(dirs.conn_eval_dir, 'model_comparison')
        io.make_dirs(fpath)

        # loop over cortical parcellations
        for cortex in df['eval_X_data'].unique():

            # grab full paths to trained models for `cortex` and filter out `methods`
            imgs = [os.path.join(dirs.conn_eval_dir, model, f'group_{metric}_vox.nii') for model in df['eval_name'] if cortex in model] 
            imgs = [img for img in imgs if any(k in img for k in methods)]

            # get binarized difference map
            nib_obj = nio.binarize_vol(imgs, metric='max')

            # save to disk
            colormap = cm.get_cmap('tab10', len(methods)+1)
            label_names = ['label-01']
            label_names.extend(methods)
            colormap.colors[0] = [0,0,0,1] # assign zero label

            # save nifti
            nib.save(nib_obj, os.path.join(fpath, f'group_difference_binary_{metric}_{cortex}.nii'))

            # map volume to surface
            surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT", stats='mode')
            gii_img = flatmap.make_label_gifti(data=surf_data, label_names=label_names, label_RGBA=colormap.colors)

            nib.save(gii_img, os.path.join(fpath, f'group_difference_binary_{metric}_{cortex}.label.gii'))

def run_subtract(
    glm='glm7', 
    metric='R', 
    methods=['ridge', 'WTA']
    ):
    """
    Args: 
        glm (str): 'glm7'
        metric (str): evaluation metric: 'R' or 'R2'. default is 'R'
        methods (list of str): default is ['WTA', 'ridge']
    Returns: 
        saves nifti and gifti for difference map between `methods` for evaluated models
    """
    # loop over experiments
    for exp in range(2):
        
        df = summary.eval_summary(exps=[f"sc{exp+1}"])
        df = df[['eval_name', 'eval_X_data']].drop_duplicates() # get unique model names

        dirs = const.Dirs(exp_name=f"sc{exp+1}", glm=glm)
        fpath = os.path.join(dirs.conn_eval_dir, 'model_comparison')
        io.make_dirs(fpath)

        # loop over cortical parcellations
        for cortex in df['eval_X_data'].unique():

            # grab full paths to trained models for `cortex` and filter out `methods`
            imgs=[]
            for method in methods:
                img = [os.path.join(dirs.conn_eval_dir, model, f'group_{metric}_vox.nii') for model in df['eval_name'] if cortex and method in model][0]
                imgs.append(img)

            # make and save differene map
            nib_obj = nio.subtract_vol(imgs)
            fname = '_'.join(methods)
            nib.save(nib_obj, os.path.join(fpath, f'group_difference_subtract_{fname}_{metric}_{cortex}.nii'))

            # map volume to surface
            surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT", stats='nanmean')
            
            # make and save gifti image
            gii_img = flatmap.make_func_gifti(data=surf_data)
            nib.save(gii_img, os.path.join(fpath, f'group_difference_subtract_{fname}_{metric}_{cortex}.func.gii'))

def subtract_acc(cortex, exp = 'sc2', normalized = True, save = True):
    plt.figure(figsize=(10,10))

    # get the name of the evaluation map
    eval_names = eval_map_names(cortex,exp = exp, methods=['WTA','ridge'])
    dirs = const.Dirs(exp_name='sc2')

    nib_acc_list = []
    nib_ceil_list = []
    V_acc_list  = []
    V_ceil_list  = []
    for m in range(2): 

        filename = os.path.join(dirs.conn_eval_dir, eval_names[m], 'group_R_vox.nii')
        print(filename)
        nib_acc_list.append(nib.load(filename))
        # VMAP[m,:]=vmap[m].agg_data()
        V_acc_list.append(nib_acc_list[m].get_fdata())

        filename = os.path.join(dirs.conn_eval_dir, eval_names[m],
                            'group_noiseceiling_XY_R_vox.nii')
        nib_ceil_list.append(nib.load(filename))
        # VCEIL[m,:]=vceil[m].agg_data()
        V_ceil_list.append(nib_ceil_list[m].get_fdata())

    if normalized: # normalize by model noise ceiling
        diff_vol = (V_acc_list[1]/V_ceil_list[1]) - (V_acc_list[0]/V_ceil_list[0])
    else:
        diff_vol = (V_acc_list[1]) - (V_acc_list[0])

    affine_mat = nib_acc_list[0].affine # to create nib object

    nib_obj = nib.Nifti1Image(diff_vol, affine_mat)
    # convert to surf
    diff_flat = flatmap.vol_to_surf(nib_obj)
    flatmap.plot(diff_flat,cscale=[-0.2,0.2], colorbar = True)
    if save:
        plt.savefig(os.path.join(dirs.figure, f'group_R_{cortex}_ridge-WTA_norm_{normalized}.svg'), dpi=300, format='svg', bbox_inches='tight', pad_inches=0)

def run():
    # run_binarize()
    run_subtract()

if __name__ == "__main__":
    run()