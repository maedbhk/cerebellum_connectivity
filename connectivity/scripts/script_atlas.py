import click
import numpy as np
import os
import connectivity.constants as const
import connectivity.io as cio
from connectivity import data as cdata
import connectivity.nib_utils as nio
from connectivity import atlas

@click.command()
@click.option("--glm")
@click.option("--atlas_name")

def run(glm, atlas_name):
    """Computes a wta map for the cerebellum based on a cortical `atlas_name`

    Each cerebellar parcellation is derived using both `sc1` and `sc2` datasets

    Args: 
        glm (str): 'glm7'
        atlas_name (str): 'yeo7', 'yeo17' etc. any cortical atlas from `data/sc1/RegionOfInterest/data/group`

    Returns: 
        saves nifti and gifti for new cerebellar atlas to `data/cerebellar_atlases`
    """
    labels = {}
    # loop over exp
    for exp in ['sc1', 'sc2']:
        labels[exp] = atlas.model_wta(['s02', 's03'], exp, glm, atlas_name) # const.return_subjs

    # concat labels across exps
    labels_concat = np.concatenate((labels['sc1'], labels['sc2']))

    # save maps to disk for cerebellum and cortex
    dirs = const.Dirs()
    fpath = os.path.join(dirs.base_dir, 'cerebellar_atlases')
    cio.make_dirs(fpath)

    # get label colors
    rgba, _ = nio.get_label_colors(fpath=os.path.join(dirs.reg_dir, 'data', 'group', f'{atlas_name}.R.label.gii'))

    nio.make_gifti_cerebellum(data=labels_concat, 
                        mask=cdata.read_mask(),
                        outpath=os.path.join(fpath, f'{atlas_name}_wta_suit.label.gii'),
                        stats='mode',
                        data_type='label',
                        save_nifti=True,
                        label_RGBA=rgba)
                        
if __name__ == "__main__":
    run()
