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
    models, cortex_names = summary.get_best_models(method=method) 

    # cortex = 'tessels1002'; models = [f'{method}_{cortex}_alpha_-2']; cortex_names = ['tessels1002']

    data_voxels_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):

        data_voxels = cweights.cortical_surface_voxels(model_name=best_model,
                                    cortex=cortex, 
                                    train_exp=exp,
                                    weights=weights,
                                    save_maps=save_maps)

        for k,v in data_voxels.items():
            data_voxels_all[k].extend(v)

    # save dataframe to disk
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_surface_voxels_stats.csv')
    df = pd.DataFrame.from_dict(data_voxels_all)  
    # if os.path.isfile(fpath):
    #     df_exist = pd.read_csv(fpath) 
    #     df = pd.concat([df_exist, df])
    df.to_csv(fpath)

def surfaces_rois(
    exp='sc1',
    weights='nonzero', 
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
    models, cortex_names = summary.get_best_models(method=method) 

    # models = [f'{method}_tessels1002_alpha_-2']; cortex_names = ['tessels1002']

    data_rois_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):
        
        
        alpha = int(best_model.split('_')[-1])
        data_rois = cweights.cortical_surface_rois(model_name=best_model, 
                                    train_exp=exp,
                                    weights=weights,
                                    alpha=alpha,
                                    cortex=cortex
                                    )
        for k,v in data_rois.items():
            data_rois_all[k].extend(v)

    # save dataframe to disk
    df = pd.DataFrame.from_dict(data_rois_all)
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_surface_rois_stats.csv')  
    # if os.path.isfile(fpath):
    #     df_exist = pd.read_csv(fpath) 
    #     df = pd.concat([df_exist, df])
    df.to_csv(fpath)

@click.command()
@click.option("--exp")
@click.option("--weights")
@click.option("--method")
@click.option("--regions")

def run(exp='sc1', 
    weights='nonzero', 
    method='lasso', 
    regions='voxels'
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
        surfaces_rois(exp=exp, weights=weights, method=method)

if __name__ == "__main__":
    run()