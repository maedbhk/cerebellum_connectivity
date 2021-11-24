import click
import os
import pandas as pd
import nibabel as nib
import numpy as np
import deepdish as dd

from connectivity import weights as cweights
from connectivity import visualize as summary
import connectivity.constants as const

@click.command()
@click.option("--atlas")
@click.option("--method")
@click.option("--exp")

def cortical_weight_maps(
    atlas='MDTB10', 
    method='ridge', 
    weights='nonzero',
    exp='sc1',
    threshold=100
    ):
    """Create cortical maps of average weights from `atlas`-based model

    Fit models using `method` for cerebellar regions defined by `atlas` and saves corresponding cortical maps
    
    Args: 
        atlas (str): default is 'MDTB10'
        method (str): default is 'ridge'. other option: 'lasso'
        weights (str): default is 'nonzero'. other option: 'positive'
        exp (str): default is 'sc1'
        threshold (int): default is 100 (i.e., no threshold)
    Returns: 
        saves cortical weight maps (*.func.gii) to disk for left and right hemispheres
    """

    dirs = const.Dirs(exp_name=exp)

    models, cortex_names = summary.get_best_models(method=method) 

    for (best_model, cortex) in zip(models, cortex_names):
        
        # full path to best model
        fpath = os.path.join(dirs.conn_train_dir, best_model)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        roi_betas_all = []
        for subj in const.return_subjs:
            roi_betas, reg_names, colors = cweights.average_region_data(subj,
                                    exp=exp, cortex=cortex, 
                                    atlas=atlas, method=method, alpha=alpha, 
                                    weights=weights, average_subjs=False)
                                    
            roi_betas_all.append(roi_betas)

        roi_betas_group = np.nanmean(np.stack(roi_betas_all), axis=0)
        giis = cweights.regions_cortex(roi_betas_group, reg_names, cortex=cortex, threshold=threshold)
            
        fname = f'group_{atlas}_threshold_{threshold}'
        [nib.save(gii, os.path.join(fpath, f'{fname}.{hem}.func.gii')) for (gii, hem) in zip(giis, ['L', 'R'])]

def run(atlas, method, exp):
    cortical_weight_maps(atlas, method, exp)

if __name__ == "__main__":
    run()