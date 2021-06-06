import click
import numpy as np
import pandas as pd
import os
import connectivity.constants as const
import connectivity.nib_utils as nio
from connectivity import visualize as summary

@click.command()
@click.option("--glm")
@click.option("--atlas")

def run(glm, atlas, metric='R'):
    """
    Args: 
        glm (str): 'glm7'
        atlas (str): 'yeo7', 'yeo17' etc. any cortical atlas from `data/sc1/RegionOfInterest/data/group`

    Returns: 
        saves nifti and gifti for new cerebellar atlas to `data/cerebellar_atlases`
    """
    methods = ['WTA', 'ridge']
    for exp in range(2):

        dirs = const.Dirs(exp_name=f"sc{exp+1}")

        # get evaluated models
        df = pd.read_csv(os.path.join(dirs.conn_eval_dir, 'eval_summary.csv'))
        df = df[['name', 'X_data']].drop_duplicates() # get unique model names

        # loop over cortical parcellations
        for cortex in df['X_data']:

            # grab full paths to models for `cortex`
            dirs = const.Dirs(exp_name=f"sc{2-exp}")
            imgs = [os.path.join(dirs.conn_eval_dir, model, f'group_{metric}_vox.nii') for model in df['name'] if cortex in model]

            # get difference map
            nio.binarize_vol(imgs, mask, metric='max')
    
    # labels = {}
    # # loop over exp
    # for exp in ['sc1', 'sc2']:
    #     labels[exp] = make_atlas.model_wta(const.return_subjs, exp, glm, atlas)

    # # concat labels across exps
    # labels_concat = np.concatenate((labels['sc1'], labels['sc2']))

    # # save maps to disk for cerebellum and cortex
    # dirs = const.Dirs()
    # fpath = os.path.join(dirs.base_dir, 'cerebellar_atlases')
    # cio.make_dirs(fpath)

    # # get label colors
    # rgba, _ = nio.get_label_colors(fpath=os.path.join(dirs.reg_dir, 'data', 'group', f'{atlas}.R.label.gii'))

    # make_atlas.save_maps_cerebellum(data=labels_concat, 
    #                     fpath=os.path.join(fpath, f'{atlas}_wta_suit'),
    #                     group='mode',
    #                     nifti=True,
    #                     label_RGBA=rgba)
                        
if __name__ == "__main__":
    run()
