from collections import defaultdict
import click
import pandas as pd
import os

from connectivity import weights as cweights
from connectivity import visualize as summary
import connectivity.constants as const

def surfaces_voxels(
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
    if not os.path.isfile(fpath):
        df_exist = pd.read_csv(fpath)
        df = pd.concat([df, df_exist])
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
    models = [m for m in models if 'mdtb' not in m]
    cortex_names = [c for c in cortex_names if 'mdtb' not in c]

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
    if not os.path.isfile(fpath):
        df_exist = pd.read_csv(fpath)
        df = pd.concat([df, df_exist])  
    df.to_csv(fpath)

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