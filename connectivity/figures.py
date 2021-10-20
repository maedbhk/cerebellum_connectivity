
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
    atlases=['yeo7'], 
    structure='cortex',
    format='png',
    colorbar=False
    ):
    """makes png containing other pngs

    Args: 
        atlases (list of str): default is ['yeo7']
        structure (str): default is 'cortex'
        format (str): default is 'png'
    """
    dirs = const.Dirs()

    if structure=='cortex':
        fpaths = nio.get_cortical_atlases(atlas_keys=atlases)
    elif structure=='cerebellum':
        fpaths = nio.get_cerebellar_atlases(atlas_keys=atlases)
    
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
    
    outpath = os.path.join(dirs.figure, ('-').join(atlases), '.png')
    img = vis.join_png(fpaths=png_paths, outpath=outpath)

    return img

def fig1(format='png'):
    plt.clf()

    vis.plotting_style()
    labelsize = 40

    dirs = const.Dirs()
    A = os.path.join(dirs.figure, f'yeo7-cortex.png')
    if not os.path.isfile(A):
        fig = vis.view_atlas_cortex(atlas='yeo7')

    A2 = os.path.join(dirs.figure, f'Buckner7-yeo7_wta-cerebellum.png')
    if not os.path.isfile(A2):
        # make_image(atlases=['Buckner7', 'yeo7_wta'], structure='cerebellum', colorbar=False)
        dirs = const.Dirs()
        img1 = os.path.join(dirs.figure, 'atl-Buckner7_sp-SUIT-label-cerebellum.png')
        img2 = os.path.join(dirs.figure, 'yeo7_wta_suit-label-cerebellum.png')
        vis.join_png(fpaths=[img1,img2], outpath=os.path.join(dirs.figure, 'Buckner7-yeo7_wta-cerebellum.png'))

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.02

    ax1 = fig.add_subplot(gs[0, 0])
    vis.plot_png(A, ax=ax1)
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, 0])
    vis.plot_png(A2, ax=ax2)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0,1])
    dataframe = vis.eval_summary(exps=['sc2'])
    vis.plot_eval_predictions(dataframe=dataframe, exps=['sc2'], methods=['WTA'], hue=None, noiseceiling=True, save=True)
    ax3.text(x_pos, y_pos+.05, 'B', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    ax3.set_xticks([80, 304, 670, 1190, 1848])

    ax4 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_WTA_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", model_name='best_model', method='WTA', save=True) # ax=ax4
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos+.05, 'C', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, f'fig1.{format}'), bbox_inches="tight", dpi=300)

def fig2(format='png'):
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(2, 3, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    dataframe = vis.train_summary(exps=['sc1'])
    ax1 = fig.add_subplot(gs[0,0])
    vis.plot_train_predictions(dataframe=dataframe, x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['tessels'], methods=['ridge'], ax=ax1)
    ax1.set_xlabel('Hyperparameter')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')
    ax1.set_ylim([.05, .4])

    ax2 = fig.add_subplot(gs[0,1])
    vis.plot_train_predictions(dataframe=dataframe.query('train_hyperparameter>-5'), x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['tessels'], methods=['lasso'], ax=ax2)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')
    ax2.set_ylim([.05, .4])
    
    ax3 = fig.add_subplot(gs[0,2])
    dataframe = vis.eval_summary(exps=['sc2'])
    vis.plot_eval_predictions(dataframe=dataframe, exps=['sc2'], methods=['WTA', 'ridge', 'lasso'], hue='eval_model', noiseceiling=True, ax=ax3)
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    ax3.set_xticks([80, 304, 670, 1190, 1848])
    
    ax4 = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'map_R_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", model_name='best_model', method='ridge', save=True);
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    ax5 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_lasso_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", model_name='best_model', method='lasso', save=True, cscale=[0, 0.5]); # ax=ax4
    vis.plot_png(fpath, ax=ax5)
    ax5.axis('off')
    ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'map_noiseceiling_Y_R_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="noiseceiling_Y_R", exp="sc1", model_name='best_model', method='ridge', save=True); # ax=ax4
    vis.plot_png(fpath, ax=ax6)
    ax6.axis('off')
    ax6.text(x_pos, y_pos, 'F', transform=ax6.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig2.{format}')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig3(format='png'):
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(3, 3, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30


