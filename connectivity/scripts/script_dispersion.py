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

# @click.command()
# @click.option("--atlas")
# @click.option("--method")
# @click.option("--exp")

def dispersion_summary(
    atlas='MDTB10', 
    method='ridge', # L2regression
    exp='sc1',
    ):

    df = pd.DataFrame()         # Empty data frame to start with
    dirs = const.Dirs(exp_name=exp)
    subjs, _ = cweights.split_subjects(const.return_subjs, test_size=0.3)

    # models, cortex_names = summary.get_best_models(method=method) 

    models = ['ridge_tessels1002_alpha_8']
    cortex_names = ['tessels1002']

    data_dict_all = defaultdict(list)
    for (best_model, cortex) in zip(models, cortex_names):

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        for subj in subjs:
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
                df = pd.concat([df,df_res])
        pass
    # save dataframe to disk
    fpath = os.path.join(dirs.conn_train_dir, 'cortical_dispersion_stats.csv')  
    df.to_csv(fpath)

def run():
    dispersion_summary()

if __name__ == "__main__":
     run()