
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import numpy as np
import os
import pandas as pd
from pathlib import Path

from connectivity import visualize as vis
from connectivity import nib_utils as nio
import connectivity.constants as const 

def make_image(
    outpath,
    atlases=['yeo7'], 
    structure='cortex',
    format='png',
    colorbar=True
    ):
    """makes png containing other pngs

    Args: 
        outpath (str): full path to saved file on disk
        atlases (list of str): default is ['yeo7']
        structure (str): default is 'cortex'
        format (str): default is 'png'
    """
    dirs = const.Dirs()

    if structure=='cortex':
        fpaths, _ = nio.get_cortical_atlases(atlas_keys=atlases)
    elif structure=='cerebellum':
        fpaths, _ = nio.get_cerebellar_atlases(atlas_keys=atlases)
    
    png_paths = []
    for fpath in fpaths:
        fname = '-'.join(Path(fpath).stem.split('.')[:2])
        atlas_path = os.path.join(dirs.figure, f'{fname}-{structure}.{format}')
        if not os.path.isfile(atlas_path):
            vis.map_atlas(fpath, structure=structure, outpath=atlas_path, colorbar=False)
        png_paths.append(atlas_path)
    
    if colorbar:
        # make colorbar separately
        fname = Path(fpath).stem.split('.')[0]
        colorbar_path = os.path.join(dirs.figure, f'{fname}-colorbar.png')
        if not os.path.isfile(colorbar_path):
            nio.view_colorbar(fpath=fpath, outpath=colorbar_path)
        png_paths.append(colorbar_path)
    
    img = vis.join_png(fpaths=png_paths, outpath=outpath)

    return img

def fig1():
    plt.clf()

    vis.plotting_style()

    dirs = const.Dirs()
    A = os.path.join(dirs.figure, f'yeo7-cortex.png')
    if not os.path.isfile(A):
        make_image(A, atlases=['yeo7'], structure='cortex', colorbar=False)

    A2 = os.path.join(dirs.figure, f'yeo7-cerebellum.png')
    if not os.path.isfile(A2):
        make_image(A2, atlases=['Buckner7', 'yeo7_wta'], structure='cerebellum', colorbar=True)

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.02

    ax1 = fig.add_subplot(gs[0, 0])
    vis.plot_png(A, ax=ax1)
    ax1.text(x_pos, 1.0, 'A', transform=ax1.transAxes, fontsize=40, verticalalignment='top')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    vis.plot_png(A2, ax=ax2)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1,0])
    vis.plot_eval_predictions(exps=['sc2'], methods=['WTA'], hue=None, noiseceiling=True, ax=ax3)
    ax3.text(x_pos, 1.2, 'B', transform=ax3.transAxes, fontsize=40, verticalalignment='top')

    # plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, 'fig1.png'), bbox_inches="tight", dpi=300)
