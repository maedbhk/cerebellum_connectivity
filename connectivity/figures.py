
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

def fig2():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(2, 3, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax1 = fig.add_subplot(gs[0,0])
    df = vis.get_summary("train",exps='sc1', method=['ridge'], atlas=['tessels'], summary_name=[''])
    vis.plot_train_predictions(df, x='hyperparameter', hue='num_regions', ax=ax1)
    ax1.set_xlabel('Hyperparameter')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')
    ax1.set_ylim([.05, .4])

    ax2 = fig.add_subplot(gs[0,1])
    df = vis.get_summary("train", exps=['sc1'], method=['lasso'], atlas=['tessels'], summary_name=[''])
    vis.plot_train_predictions(df, x='hyperparameter', hue='num_regions', ax=ax2)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')
    ax2.set_ylim([.05, .4])
    
    ax3 = fig.add_subplot(gs[0,2])
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['tessels'], method=['WTA', 'ridge', 'lasso'], summary_name=['weighted_all'])
    vis.plot_eval_predictions(dataframe=dataframe, plot_noiseceiling=True, normalize=False, hue='method', ax=ax3)
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    ax3.set_xticks([80, 304, 670, 1190, 1848])

    ax4 = fig.add_subplot(gs[1,0])
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['tessels'], method=['WTA', 'ridge', 'lasso'], summary_name=['weighted_all'])
    vis.plot_eval_predictions(dataframe=dataframe, plot_noiseceiling=False, normalize=True, hue='method', ax=ax4)
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')
    ax4.set_xticks([80, 304, 670, 1190, 1848])
    
    ax5 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_ridge_best_model_normalize.png')
    best_model = 'ridge_tessels1002_alpha_8'
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", model_name=best_model, normalize=True, method='ridge', outpath=fpath); # cscale=[0, 0.4]
    vis.plot_png(fpath, ax=ax5)
    ax5.axis('off')
    ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    # ax5 = fig.add_subplot(gs[1,1])
    # fpath = os.path.join(dirs.figure, f'map_R_lasso_best_model.png')
    # best_model = 'lasso_tessels1002_alpha_-2'
    # if not os.path.isfile(fpath):
    #     vis.map_eval_cerebellum(data="R", model_name=best_model, method='lasso', cscale=[0, 0.4], outpath=fpath); # ax=ax4
    # vis.plot_png(fpath, ax=ax5)
    # ax5.axis('off')
    # ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'map_noiseceiling_XY_R_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="noiseceiling_XY_R", normalize=False, model_name='best_model', method='ridge', outpath=fpath); # ax=ax4
    vis.plot_png(fpath, ax=ax6)
    ax6.axis('off')
    ax6.text(x_pos, y_pos, 'F', transform=ax6.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig2.{format}')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig3():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(5, 4, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax1 = fig.add_subplot(gs[0,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=0, outpath=fpath)
    vis.plot_png(fpath, ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')

    ax2 = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'MDTB-reg1.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[1], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax2)
    ax2.text(x_pos, y_pos, '', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg2.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[2], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax3)
    ax3.text(x_pos, y_pos, '', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg2.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=1, outpath=fpath)
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos, 'B', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg3.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=2, outpath=fpath)
    vis.plot_png(fpath, ax=ax6)
    ax6.axis('off')
    ax6.text(x_pos, y_pos, 'C', transform=ax6.transAxes, fontsize=labelsize, verticalalignment='top')

    ax7 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'MDTB-reg3.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[3], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax7)
    ax7.text(x_pos, y_pos, '', transform=ax7.transAxes, fontsize=labelsize, verticalalignment='top')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg4.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[4], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax8)
    ax8.text(x_pos, y_pos, '', transform=ax8.transAxes, fontsize=labelsize, verticalalignment='top')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[1,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg4.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=3, outpath=fpath)
    vis.plot_png(fpath, ax=ax9)
    ax9.axis('off')
    ax9.text(x_pos, y_pos, 'D', transform=ax9.transAxes, fontsize=labelsize, verticalalignment='top')

    ax10 = fig.add_subplot(gs[2,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg5.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=4, outpath=fpath)
    vis.plot_png(fpath, ax=ax10)
    ax10.axis('off')
    ax10.text(x_pos, y_pos, 'E', transform=ax10.transAxes, fontsize=labelsize, verticalalignment='top')

    ax11 = fig.add_subplot(gs[2,1])
    fpath = os.path.join(dirs.figure, f'MDTB-reg5.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[5], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax11)
    ax11.text(x_pos, y_pos, '', transform=ax11.transAxes, fontsize=labelsize, verticalalignment='top')
    ax11.axis('off')

    ax12 = fig.add_subplot(gs[2,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg6.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[6], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax12)
    ax12.text(x_pos, y_pos, '', transform=ax12.transAxes, fontsize=labelsize, verticalalignment='top')
    ax12.axis('off')

    ax13 = fig.add_subplot(gs[2,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg6.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=5, outpath=fpath)
    vis.plot_png(fpath, ax=ax13)
    ax13.axis('off')
    ax13.text(x_pos, y_pos, 'F', transform=ax13.transAxes, fontsize=labelsize, verticalalignment='top')

    ax14 = fig.add_subplot(gs[3,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg7.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=6, outpath=fpath)
    vis.plot_png(fpath, ax=ax14)
    ax14.axis('off')
    ax14.text(x_pos, y_pos, 'G', transform=ax14.transAxes, fontsize=labelsize, verticalalignment='top')

    ax15 = fig.add_subplot(gs[3,1])
    fpath = os.path.join(dirs.figure, f'MDTB-reg7.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[7], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax15)
    ax15.text(x_pos, y_pos, '', transform=ax15.transAxes, fontsize=labelsize, verticalalignment='top')
    ax15.axis('off')

    ax16 = fig.add_subplot(gs[3,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg8.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[8], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax16)
    ax16.text(x_pos, y_pos, '', transform=ax16.transAxes, fontsize=labelsize, verticalalignment='top')
    ax16.axis('off')

    ax17 = fig.add_subplot(gs[3,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg8.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=7, outpath=fpath)
    vis.plot_png(fpath, ax=ax17)
    ax17.axis('off')
    ax17.text(x_pos, y_pos, 'H', transform=ax17.transAxes, fontsize=labelsize, verticalalignment='top')

    ax18 = fig.add_subplot(gs[4,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg9.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=8, outpath=fpath)
    vis.plot_png(fpath, ax=ax18)
    ax18.axis('off')
    ax18.text(x_pos, y_pos, 'I', transform=ax18.transAxes, fontsize=labelsize, verticalalignment='top')

    ax19 = fig.add_subplot(gs[4,1])
    fpath = os.path.join(dirs.figure, f'MDTB-reg9.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[9], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax19)
    ax19.text(x_pos, y_pos, '', transform=ax19.transAxes, fontsize=labelsize, verticalalignment='top')
    ax19.axis('off')

    ax20 = fig.add_subplot(gs[4,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg10.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[10], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax20)
    ax20.text(x_pos, y_pos, '', transform=ax20.transAxes, fontsize=labelsize, verticalalignment='top')
    ax20.axis('off')

    ax21 = fig.add_subplot(gs[4,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg10.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=9, outpath=fpath)
    vis.plot_png(fpath, ax=ax21)
    ax21.axis('off')
    ax21.text(x_pos, y_pos, 'J', transform=ax21.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig3.png')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig4():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(2, 3, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax1 = fig.add_subplot(gs[:,0])
    fpath = os.path.join(dirs.figure, f'group_lasso_percent_nonzero_cerebellum.png')
    if not os.path.isfile(fpath):
        vis.map_lasso_cerebellum(model_name='lasso_tessels0362_alpha_-2', stat='percent', weights='nonzero', outpath=fpath);
    vis.plot_png(fpath, ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')

    ax2 = fig.add_subplot(gs[0,1:])
    ax2,_ = vis.plot_surfaces(x='reg_names', hue=None, cortex='tessels0042', method='lasso', regions=None, ax=ax2);
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')

    ax3 = fig.add_subplot(gs[1,1:])
    ax3,_ = vis.plot_dispersion(y='var_w', hue='hem', cortex='tessels0042', atlas='MDTB10',regions=None, ax=ax3);
    plt.ylim([0.58, .83]);
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig4.png')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig5(format='png'):
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax1 = fig.add_subplot(gs[0,0])
    vis.plot_test_predictions(ax=ax1, hue='test_routine')
    ax1.set_xticks([80, 304, 670, 1190, 1848])
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')

    ax2 = fig.add_subplot(gs[0,1])

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, f'fig4.{format}'), bbox_inches="tight", dpi=300)

def figS1(format='png'):
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
    vis.plot_train_predictions(dataframe=dataframe, x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['schaefer'], methods=['ridge'], ax=ax1)
    ax1.set_xlabel('Hyperparameter')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')
    ax1.set_ylim([.05, .4])

    ax2 = fig.add_subplot(gs[0,1])
    vis.plot_train_predictions(dataframe=dataframe.query('train_hyperparameter>-5'), x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['schaefer'], methods=['lasso'], ax=ax2)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')
    ax2.set_ylim([.05, .4])
    
    ax3 = fig.add_subplot(gs[0,2])
    dataframe = vis.eval_summary(exps=['sc2'])
    vis.plot_predictions(dataframe=dataframe, exps=['sc2'], methods=['WTA', 'ridge'], atlases=['schaefer'], hue='model',  ax=ax3) # 'lasso'
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    
    ax4 = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'map_R_ridge_schaefer_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='schaefer', model_name='best_model', method='ridge',  cscale=[0, 0.4], outpath=fpath);
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    ax5 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_lasso_schaefer_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='schaefer', model_name='best_model', method='lasso',  cscale=[0, 0.4], outpath=fpath); # ax=ax4
    vis.plot_png(fpath, ax=ax5)
    ax5.axis('off')
    ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'map_R_WTA_schaefer_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='schaefer', model_name='best_model', method='WTA',  cscale=[0, 0.4], outpath=fpath) # ax=ax4
    vis.plot_png(fpath, ax=ax6)
    ax6.axis('off')
    ax6.text(x_pos, y_pos+.05, 'C', transform=ax6.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS2.{format}')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS2(format='png'):
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(2, 4, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax2 = fig.add_subplot(gs[0,0])
    vis.plot_surfaces(y='percent', cortex_group='tessels', weights='nonzero', regions=['Region3', 'Region6', 'Region7', 'Region8', 'Region9', 'Region10'], hue='reg_names', method='lasso', ax=ax2);
    ax2.set_xticks([80, 304, 670, 1190, 1848])
    ax2.text(x_pos, y_pos, 'A', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')

    ax3 = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg3.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=2, outpath=fpath)
    vis.plot_png(fpath, ax=ax3)
    ax3.axis('off')
    ax3.text(x_pos, y_pos, 'B', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')

    ax4 = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg6.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=5, outpath=fpath)
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos, 'C', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    ax5 = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg7.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=6, outpath=fpath)
    vis.plot_png(fpath, ax=ax5)
    ax5.axis('off')
    ax5.text(x_pos, y_pos, 'D', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg8.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=7, outpath=fpath)
    vis.plot_png(fpath, ax=ax6)
    ax6.axis('off')
    ax6.text(x_pos, y_pos, 'E', transform=ax6.transAxes, fontsize=labelsize, verticalalignment='top')

    ax7 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg9.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=8, outpath=fpath)
    vis.plot_png(fpath, ax=ax7)
    ax7.axis('off')
    ax7.text(x_pos, y_pos, 'F', transform=ax7.transAxes, fontsize=labelsize, verticalalignment='top')

    ax8 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg10.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=9, outpath=fpath)
    vis.plot_png(fpath, ax=ax8)
    ax8.axis('off')
    ax8.text(x_pos, y_pos, 'G', transform=ax8.transAxes, fontsize=labelsize, verticalalignment='top')

    ax9 = fig.add_subplot(gs[1,3])
    vis.plot_dispersion(hue='hem', y='Variance', cortex_group='tessels', cortex='tessels1002', atlas='MDTB10', regions=[3,6,7,8,9,10], ax=ax9)
    ax9.text(x_pos, y_pos, 'H', transform=ax9.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS3.{format}')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
