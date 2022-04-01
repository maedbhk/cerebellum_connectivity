from collections import defaultdict
import click
import pandas as pd
import numpy as np
import os

from connectivity import weights as cweights
from connectivity import visualize as summary
import connectivity.constants as const
import connectivity.data as cdata

def surfaces_voxels_depricated(
    exp='sc1',
    weights='nonzero', 
    method='lasso',
    save_maps=True, 
    ):
    """compute surface maps for cerebellum (count # of non-zero coefs for each cerebellar voxel)

    Args: 
        exp (str): default is 'sc1'
        weights (str): default is 'nonzero'. other option: 'positive'
        method (str): default is 'lasso', other option is 'L2regression'
    Returns: 
        saves cerebellar gifti to disk and summary csv
    """

    print('running surfaces_voxels')

    dirs = const.Dirs(exp_name=exp)
    dataframe = summary.get_summary(exps=[exp], summary_type='train', method=[method])
    models, cortex_names= summary.get_best_models(dataframe)
    models = [m for m in models if 'tessels' in m]
    cortex_names = [c for c in cortex_names if 'tessels' in c]

    data_voxels_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):

        data_voxels = cweights.cortical_surface_voxels(model_name=best_model,
                                    cortex=cortex, 
                                    train_exp=exp,
                                    method=method,
                                    weights=weights,
                                    save_maps=save_maps)

        for k,v in data_voxels.items():
            data_voxels_all[k].extend(v)

    # save dataframe to disk
    df = pd.DataFrame.from_dict(data_voxels_all)
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_surface_voxels_stats.csv')
    if os.path.isfile(fpath):
        df = pd.concat([df, pd.read_csv(fpath)])
    df.to_csv(fpath)

def surfaces_rois(
    exp='sc1',
    weights='nonzero', 
    atlas='MDTB10',
    method='lasso', # L2regression
    ):
    """compute summary data for cerebellar regions (count # of non-zero coefs for each cerebellar region)

    Args: 
        exp (str): default is 'sc1'
        weights (str): default is 'nonzero'. other option: 'positive'
        method (str): default is 'lasso', other option is 'L2regression'
    Returns: 
        saves summary csv to disk
    """

    print('running surfaces_rois')

    dirs = const.Dirs(exp_name=exp)
    dataframe = summary.get_summary(exps=[exp], summary_type='train', method=[method])
    models, cortex_names= summary.get_best_models(dataframe)
    models = [m for m in models if 'tessels' in m]
    cortex_names = [c for c in cortex_names if 'tessels' in c]

    data_rois_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):
        
        alpha = int(best_model.split('_')[-1])
        data_rois = cweights.cortical_surface_rois(model_name=best_model, 
                                    train_exp=exp,
                                    weights=weights,
                                    alpha=alpha,
                                    method=method,
                                    atlas=atlas,
                                    cortex=cortex
                                    )
        for k,v in data_rois.items():
            data_rois_all[k].extend(v)

    # save dataframe to disk
    df = pd.DataFrame.from_dict(data_rois_all)
    fpath = os.path.join(dirs.conn_train_dir, f'cortical_surface_rois_stats_{atlas}.csv') 
    if os.path.isfile(fpath):
        df = pd.concat([df, pd.read_csv(fpath)])  
    df.to_csv(fpath)

def surfaces_voxels(
    atlas = 'MDTB10',
    method='ridge',
    exp='sc1', 
    weights = 'nonzero',
    average_subjs = False # average weights across subjects and then calculate dispersion? 
    ):
    """compute surface maps for cerebellum (count # of non-zero coefs for each cerebellar voxel)

    Args: 
        exp (str): default is 'sc1'
        weights (str): default is 'nonzero'. other option: 'positive'
        method (str): default is 'lasso', other option is 'L2regression'
    Returns: 
        saves cerebellar gifti to disk and summary csv
    """

    print('running surfaces_voxels')

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
    data_percent_subs = []
    data_count_subs = []
    df_list = []
    for (best_model, cortex) in zip(models, cortex_names):
        print(f"surface stats for {best_model}")
        
        model_data = cweights.get_model_data(best_model, train_exp=exp, average_subjs=average_subjs)

        if not average_subjs:
            [num_subjs, _, _] = model_data.shape
        else:
            num_subjs = 1
        # loop over subjects and calculate measures for each subject
        for s in range(num_subjs):
            stats_df = pd.DataFrame()

            try:
                weight_map = model_data[s, :, :]
            except:
                weight_map = model_data
            if method=='ridge':
                weight_map = cweights._threshold_data(data=weight_map, threshold=weight_map.mean() + weight_map.std())

            df = cweights.surface_cortex(roi_betas = weight_map, weights=weights)
            subj = const.return_subjs[s]
            print(f'- {subj}')
            count = df['count'].values
            # print(count)
            percent = df['percent'].values
            # data = np.reshape(count, (1, len(count)))
            #get the average within each roi in atlas
            count_roi, _ = cdata.average_by_roi(count, index)
            percent_roi, _ = cdata.average_by_roi(percent, index)

            stats_df['subj']=[subj]*num_roi
            stats_df['roi'] = np.arange(num_roi)
            # # df_res['subj']='all-subjs'
            stats_df['cortex']=[cortex]*num_roi
            stats_df['method']=[method]*num_roi
            stats_df['atlas']=[atlas]*num_roi
            # print(w_var_roi)
            stats_df['count']=count_roi[0][1:]
            stats_df['percent']=percent_roi[0][1:]
            print(stats_df)
            df_list.append(stats_df)

            data_percent_subs.append(df['percent'].values.reshape(1, -1))
            data_count_subs.append(df['count'].values.reshape(1, -1))

        # get the average across subjects and save the map
        data_group_percent = np.concatenate(data_percent_subs, axis = 0)
        cweights.save_maps_cerebellum(np.nanmean(data_group_percent, axis=0), fpath=os.path.join(dirs.conn_train_dir, best_model, 'group_surface_percent'))
        data_group_count = np.concatenate(data_percent_subs, axis = 0)
        cweights.save_maps_cerebellum(np.nanmean(data_group_count, axis=0), fpath=os.path.join(dirs.conn_train_dir, best_model, 'group_surface_count'))

    # save dataframe to disk
    df = pd.concat(df_list)
    fpath = os.path.join(dirs.conn_train_dir, f'cortical_surface_stats_vox_{method}_{atlas}.csv') 
    df.to_csv(fpath)
    print(f'surface stats saved to disk for {atlas} and {method}')

    return

@click.command()
@click.option("--exp")
@click.option("--weights")
@click.option("--method")
@click.option("--regions")
@click.option("--atlas")

def run(exp='sc1', 
    weights='nonzero', 
    method='lasso', 
    regions='voxels',
    atlas='MDTB10'
    ):
    """run surfaces

    Args: 
        exp (str): default is 'sc1'
        weights (str): default is 'nonzero'
        method (str): default is 'lasso'
        regions (str): 'voxels' or 'rois'
    """
    if regions=='voxels':
        surfaces_voxels(exp=exp, weights=weights, method=method)
    elif regions=='rois':
        surfaces_rois(exp=exp, weights=weights, method=method, atlas=atlas)

if __name__ == "__main__":
    run()