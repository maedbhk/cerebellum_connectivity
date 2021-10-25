from collections import defaultdict
import click
import os
import pandas as pd
import nibabel as nib
import numpy as np
import deepdish as dd

from connectivity import weights as cweights
from connectivity import visualize as summary
import connectivity.constants as const

def distances_summary(
    atlas='MDTB10', 
    weights='nonzero', 
    method='ridge', # L2regression
    thresholds=[1,5],
    metric='gmean',
    exp='sc1',
    ):
    """Computes summary of cortical distances for `atlas`-based models
    
    Fit models using `method` for cerebellar regions defined by `atlas`. 
    Thresholds distances using `thresholds` for each cerebellar region

    Args: 
        atlas (str): default is 'MDTB10'
        weights (str): default is 'nonzero'. other option: 'positive'
        method (str): default is 'ridge'. other option: 'lasso'
        thresholds (list of int): default is [1,5]
        metric (str): default is 'gmean'
        exp (str): default is 'sc1'
    Returns: 
        dataframe of distances (shape; n_cerebellar_regs,)
    """


    dirs = const.Dirs(exp_name=exp)
    subjs, _ = cweights.split_subjects(const.return_subjs, test_size=0.3)

    # models, cortex_names = summary.get_best_models(method=method) 
    models = [f'{method}_tessels1002_alpha_8']; cortex_names = ['tessels1002']

    data_dict_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        for threshold in thresholds:
            for subj in subjs:
                roi_betas, reg_names, colors = cweights.average_region_data(subj,
                                        exp=exp, cortex=cortex, 
                                        atlas=atlas, method=method, alpha=alpha, 
                                        weights=weights, average_subjs=False)

                # save out cortical distances
                data_dict = cweights.distances_cortex(roi_betas, reg_names, colors, 
                            cortex=cortex, threshold=threshold, metric=metric)
                data_dict.update({'subj': np.repeat(subj, len(reg_names)*2)})

                for k, v in data_dict.items():
                    data_dict_all[k].extend(v)

    # save dataframe to disk
    df = pd.DataFrame.from_dict(data_dict_all) 
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_distances_stats.csv')  
    if os.path.isfile(fpath):
        df_exist = pd.read_csv(fpath) 
        df = pd.concat([df_exist, df])
    df.to_csv(fpath)
        
def run():
    distances_summary()

if __name__ == "__main__":
    run()