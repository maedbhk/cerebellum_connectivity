import click
import numpy as np
import os
import connectivity.constants as const
import connectivity.io as cio
import connectivity.nib_utils as nio
from connectivity import make_atlas

@click.command()
@click.option("--atlas")
@click.option("--glm")

def run(atlas, glm='glm7'):
    labels = {}
    
    # loop over exp
    for exp in ['sc1', 'sc2']:
        labels[exp] = make_atlas.model_wta(const.return_subjs, exp, glm, atlas)

    # concat labels across exps
    labels_concat = np.concatenate((labels['sc1'], labels['sc2']))

    # save maps to disk for cerebellum and cortex
    dirs = const.Dirs()
    fpath = os.path.join(dirs.base_dir, 'cerebellar_atlases')
    cio.make_dirs(fpath)

    # get label colors
    rgba, _ = nio.get_gifti_colors(fpath=os.path.join(dirs.reg_dir, 'data', 'group', f'{atlas}.R.label.gii'))

    make_atlas.save_maps_cerebellum(data=labels_concat, 
                        fpath=os.path.join(fpath, f'{atlas}_wta_suit'),
                        group='mode',
                        nifti=True,
                        label_RGBA=rgba)

if __name__ == "__main__":
    run()
