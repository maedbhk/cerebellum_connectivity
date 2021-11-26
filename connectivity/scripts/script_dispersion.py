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

def dispersion_summary(
    atlas='MDTB10', 
    method='ridge', # L2regression
    exp='sc1',
    ):

    print('running dispersion summary')

    df = pd.DataFrame()         # Empty data frame to start with
    dirs = const.Dirs(exp_name=exp)

    models, cortex_names = summary.get_best_models(method=method) 

    data_dict_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):

        if 'mdtb4002' not in cortex:

            # get alpha for each model
            alpha = int(best_model.split('_')[-1])
            for subj in const.return_subjs:
                    roi_betas, reg_names, colors = cweights.average_region_data(subj,
                                            exp=exp, cortex=cortex, 
                                            atlas=atlas, method=method, alpha=alpha, 
                                            weights='nonzero', average_subjs=False)

                    # save out cortical distances
                    df_res = cweights.dispersion_cortex(roi_betas, reg_names, colors, cortex=cortex)
                    N=df_res.shape[0]
                    df_res['subj']=[subj]*N
                    df_res['cortex']=[cortex]*N
                    df_res['method']=[method]*N
                    df_res['atlas']=[atlas]*N
                    df = pd.concat([df,df_res])
            
    # save dataframe to disk
    fpath = os.path.join(dirs.conn_train_dir, f'cortical_dispersion_stats_{atlas}.csv')  
    df.to_csv(fpath)

@click.command()
@click.option("--atlas")
@click.option("--method")
@click.option("--exp")

def run(atlas='MDTB10', 
        method='ridge', 
        exp='sc1'
        ):
    dispersion_summary(atlas=atlas, method=method, exp=exp)

if __name__ == "__main__":
     run()