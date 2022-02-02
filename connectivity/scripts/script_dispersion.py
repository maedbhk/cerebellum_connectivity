from collections import defaultdict
import click
import os
import pandas as pd
import nibabel as nib
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

from connectivity import weights as cweights
from connectivity import visualize as summary
from connectivity import data as cdata
import connectivity.constants as const

import warnings
warnings.filterwarnings('ignore')

def dispersion_rois(
    atlas='MDTB10', 
    method='ridge', # L2regression
    exp='sc1',
    ):

    print('running dispersion summary for rois')

    df = pd.DataFrame()         # Empty data frame to start with
    dirs = const.Dirs(exp_name=exp)

    dataframe = summary.get_summary(exps=[exp], summary_type='train', method=[method])
    models, cortex_names= summary.get_best_models(dataframe)
    # models = ['lasso_tessels0042_alpha_-3']
    # cortex_names = ['tessels0042']

    for (best_model, cortex) in zip(models, cortex_names):

        if 'mdtb4002' not in cortex:

            # get alpha for each model
            alpha = int(best_model.split('_')[-1])
            for subj in const.return_subjs:
                    roi_betas, _, _ = cweights.average_region_data(subj,
                                            exp=exp, cortex=cortex, 
                                            atlas=atlas, method=method, alpha=alpha, 
                                            weights='nonzero', average_subjs=False)

                    # save out cortical distances
                    df_res = cweights.dispersion_cortex(roi_betas, cortex=cortex)
                    N=df_res.shape[0]
                    df_res['subj']=[subj]*N
                    df_res['cortex']=[cortex]*N
                    df_res['method']=[method]*N
                    df_res['atlas']=[atlas]*N
                    df = pd.concat([df,df_res])
            
    # save dataframe to disk
    fpath = os.path.join(dirs.conn_train_dir, f'cortical_dispersion_stats_{atlas}.csv') 
    if os.path.isfile(fpath): 
        df_atlas = pd.read_csv(fpath)
        df = pd.concat([df, df_atlas])
    df.to_csv(fpath)
    print(f'dispersion stats saved to disk for {atlas}')

def dispersion_voxels(
    method='ridge',
    exp='sc1'
    ):

    print('running dispersion summary for voxels')

    df = pd.DataFrame()         # Empty data frame to start with
    dirs = const.Dirs(exp_name=exp)

    dataframe = summary.get_summary(exps=[exp], summary_type='train', method=[method])
    models, cortex_names= summary.get_best_models(dataframe)
    # models = ['lasso_tessels0042_alpha_-3']
    # cortex_names = ['tessels0042']

    for (best_model, cortex) in zip(models, cortex_names):
        if 'mdtb4002' not in cortex:
            model_data = cweights.get_model_data(best_model, train_exp=exp, average_subjs=True)
        
        df = cweights.dispersion_cortex(roi_betas=model_data, cortex=cortex)
        df['w_var'] = df.Variance * df.sum_w
        df['var_w'] = df.w_var / df.sum_w

        # save giftis and niftis to disk'
        data_all = np.zeros((2, int(len(df)/2)))
        for hem in [0,1]:
            df_hem = df.query(f'hem=={hem}')
            data = np.reshape(np.array(df_hem['var_w'].tolist()), (1, len(df_hem)))
            data_all[hem,:] = data
        cweights.save_maps_cerebellum(np.nanmean(data_all, axis=0), fpath=os.path.join(dirs.conn_train_dir, best_model, 'group_dispersion_var_w'))

    print(f'dispersion stats saved to disk for voxels')

@click.command()
@click.option("--atlas")
@click.option("--method")
@click.option("--regions")

def run(atlas='MDTB10', 
        method='ridge', 
        regions='rois'
        ):
    if regions=='voxels':
        dispersion_voxels(method=method)
    elif regions=='rois':
        dispersion_rois(atlas=atlas, method=method)


if __name__ == "__main__":
     run()