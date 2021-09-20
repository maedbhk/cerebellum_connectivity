# import libraries
import pandas as pd
import os
from matplotlib import cm
import nibabel as nib
import SUITPy.SUITPy.flatmap as flatmap

import connectivity.constants as const
import connectivity.nib_utils as nio
from connectivity import io 
from connectivity import visualize as summary

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

def run():
    # run_binarize()
    run_subtract()

if __name__ == "__main__":
    run()