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
import connectivity.constants as const
import connectivity.data as cdata

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
    models = [m for m in models if 'tessels' in m]
    cortex_names = [c for c in cortex_names if 'tessels' in c]

    # models = ['lasso_tessels0042_alpha_-3']
    # cortex_names = ['tessels0042']

    for (best_model, cortex) in zip(models, cortex_names):

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        if method=='lasso':
            alpha = -5
        for subj in const.return_subjs: # const.return_subjs
            roi_betas, _, _ = cweights.average_region_data(subjs=subj,# const.return_subjs
                                    exp=exp, cortex=cortex, hemispheres=True,
                                    atlas=atlas, method=method, alpha=alpha, 
                                    weights='nonzero', average_subjs=False)

            # save out cortical distances
            df_res = cweights.dispersion_cortex(roi_betas, cortex=cortex)
            N=df_res.shape[0]
            df_res['subj']=[subj]*N
            # df_res['subj']='all-subjs'
            df_res['cortex']=[cortex]*N
            df_res['method']=[method]*N
            df_res['atlas']=[atlas]*N
            df_res['w_var']=df_res.Variance*df_res.sum_w
            df_res['var_w'] = df_res.w_var/df_res.sum_w
            df = pd.concat([df,df_res])
        
    # save dataframe to disk
    fpath = os.path.join(dirs.conn_train_dir, f'cortical_dispersion_stats_{atlas}.csv') 
    if os.path.isfile(fpath): 
        df_atlas = pd.read_csv(fpath)
        df = pd.concat([df, df_atlas])
    df.to_csv(fpath)
    print(f'dispersion stats saved to disk for {atlas} and {method}')

def dispersion_voxels(
    atlas = 'MDTB10',
    method='ridge',
    exp='sc1', 
    average_subjs = False # average weights across subjects and then calculate dispersion? 
    ):

    print('running dispersion summary for voxels')

    df = pd.DataFrame()         # Empty data frame to start with
    dirs = const.Dirs(exp_name=exp)

    dataframe = summary.get_summary(exps=[exp], summary_type='train', method=[method])
    models, cortex_names= summary.get_best_models(dataframe)
    models = [m for m in models if 'tessels' in m]
    cortex_names = [c for c in cortex_names if 'tessels' in c]

    # get the atlas file
    atlas_file = os.path.join(dirs.cerebellar_atlases, 'king_2019', f'atl-{atlas}.nii')
    # get the regions
    index = cdata.read_suit_nii(atlas_file)
    num_roi = int(np.max(index))
    data_all_subs = []
    df_list = []
    for (best_model, cortex) in zip(models, cortex_names):
        print(f"dispersion for {best_model}")
        
        model_data = cweights.get_model_data(best_model, train_exp=exp, average_subjs=average_subjs)

        if not average_subjs:
            [num_subjs, _, _] = model_data.shape
        else:
            num_subjs = 1

        # loop over subjects and calculate measures for each subject
        for s in range(num_subjs):

            try:
                weight_map = model_data[s, :, :]
            except:
                weight_map = model_data
            if method=='ridge':
                #
                weight_map = cweights._threshold_data(data=weight_map, threshold=weight_map.mean() + weight_map.std())

            df = cweights.dispersion_cortex(roi_betas = weight_map, cortex = cortex)
            df['w_var'] = df.Variance * df.sum_w
            df['var_w'] = df.w_var / df.sum_w
            subj = const.return_subjs[s]
            print(f'- {subj}')

            # save giftis and niftis to disk'
            data_all = np.zeros((2, int(len(df)/2)))
            for hem in [0,1]:
                stats_df = pd.DataFrame()
                df_hem = df.query(f'hem=={hem}')
                data_var_w = np.reshape(np.array(df_hem['var_w'].tolist()), (1, len(df_hem)))
                data_w_var = np.reshape(np.array(df_hem['w_var'].tolist()), (1, len(df_hem)))

                data_all[hem,:] = data_var_w

                #get the average within each roi in atlas
                var_w_roi, _ = cdata.average_by_roi(data_var_w, index)
                w_var_roi, _ = cdata.average_by_roi(data_w_var, index)
                stats_df['subj']=[subj]*num_roi
                stats_df['roi'] = np.arange(num_roi)
                # # df_res['subj']='all-subjs'
                stats_df['cortex']=[cortex]*num_roi
                stats_df['method']=[method]*num_roi
                stats_df['atlas']=[atlas]*num_roi
                stats_df['h'] = [hem]*num_roi
                # print(w_var_roi)
                stats_df['w_var']=w_var_roi[0][1:]
                stats_df['var_w'] = var_w_roi[0][1:]
                print(stats_df)
                df_list.append(stats_df)
            # get the average across hemisphere for the current subject
            data_all_subs.append(np.nanmean(data_all, axis=0).reshape(1, -1))

        # get the average across subjects and save the map
        data_group = np.concatenate(data_all_subs, axis = 0)
        cweights.save_maps_cerebellum(np.nanmean(data_group, axis=0), fpath=os.path.join(dirs.conn_train_dir, best_model, 'group_dispersion_var_w'))

    # save dataframe to disk
    df = pd.concat(df_list)
    fpath = os.path.join(dirs.conn_train_dir, f'cortical_dispersion_stats_vox_{method}_{atlas}.csv') 
    df.to_csv(fpath)
    print(f'dispersion stats saved to disk for {atlas} and {method}')

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