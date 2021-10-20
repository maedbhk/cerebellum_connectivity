from collections import defaultdict
import click
import pandas as pd
import os

from connectivity import weights as cweights
from connectivity import visualize as summary
import connectivity.constants as const

@click.command()
@click.option("--exp")
@click.option("--weights")
@click.option("--method")

def surfaces_voxels(
    exp='sc1',
    weights='nonzero', 
    method='lasso', # L2regression
    ):

    dirs = const.Dirs(exp_name=exp)
    models, cortex_names = summary.get_best_models(method=method) 

    # cortex = 'tessels1002'; models = [f'{method}_{cortex}_alpha_-2']; cortex_names = ['tessels1002']

    data_voxels_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):

        data_voxels = cweights.cortical_surface_voxels(model_name=best_model,
                                    cortex=cortex, 
                                    train_exp=exp,
                                    weights=weights,
                                    save_maps=False)

        for k,v in data_voxels.items():
            data_voxels_all[k].extend(v)

    # save dataframe to disk
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_surface_voxels_stats.csv')
    df = pd.DataFrame.from_dict(data_voxels_all)  
    if os.path.isfile(fpath):
        df_exist = pd.read_csv(fpath) 
        df = pd.concat([df_exist, df])
    df.to_csv(fpath)

def surfaces_rois(
    exp='sc1',
    weights='nonzero', 
    method='lasso', # L2regression
    ):

    dirs = const.Dirs(exp_name=exp)
    models, cortex_names = summary.get_best_models(method=method) 

    # cortex = 'tessels1002'; models = [f'{method}_{cortex}_alpha_-2']; cortex_names = ['tessels1002']

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
    if os.path.isfile(fpath):
        df_exist = pd.read_csv(fpath) 
        df = pd.concat([df_exist, df])
    df.to_csv(fpath)

if __name__ == "__main__":
    surfaces_voxels()
    surfaces_rois()