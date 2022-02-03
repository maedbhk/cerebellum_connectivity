import click
import os
import pandas as pd
import nibabel as nib
import numpy as np
import deepdish as dd

from connectivity import weights as cweights
from connectivity import visualize as summary
import connectivity.constants as const

import warnings
warnings.filterwarnings('ignore')

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

    print('running cortical weight maps')

    dirs = const.Dirs(exp_name=exp)

    dataframe = summary.get_summary(exps=[exp], summary_type='train', method=[method])
    models, cortex_names= summary.get_best_models(dataframe)
    models = [m for m in models if 'tessels' in m]
    cortex_names = [c for c in cortex_names if 'tessels' in c]

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

        print(f'cortical weights saved to disk for {atlas}')

@click.command()
@click.option("--atlas")
@click.option("--method")
@click.option("--exp")

def run(atlas='MDTB10', 
        method='ridge',
        exp='sc1'
        ):
    cortical_weight_maps(atlas=atlas, method=method, exp=exp)

if __name__ == "__main__":
    run()