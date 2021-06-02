import click
import numpy as np
import os
import connectivity.constants as const
import connectivity.io as cio
from connectivity import make_atlas

@click.command()
@click.option("--glm")
@click.option("--atlas")

def run(glm, atlas):
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

    rgba = make_atlas.get_label_colors(atlas)

    make_atlas.save_maps_cerebellum(data=labels_concat, 
                        fpath=os.path.join(fpath, f'{atlas}_wta_suit'),
                        group='mode',
                        nifti=True,
                        label_RGBA=rgba)
if __name__ == "__main__":
    run()
