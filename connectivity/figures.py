
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

    ax = fig.add_subplot(gs[0,0])
    df = vis.get_summary("train",exps='sc1', method=['ridge'], atlas=['tessels'], summary_name=[''])
    vis.plot_train_predictions(df, x='hyperparameter', hue='num_regions', ax=ax)
    ax.set_xlabel('Hyperparameter')
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([.05, .4])

    ax = fig.add_subplot(gs[0,1])
    df = vis.get_summary("train", exps=['sc1'], method=['lasso'], atlas=['tessels'], summary_name=[''])
    vis.plot_train_predictions(df, x='hyperparameter', hue='num_regions', ax=ax)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([.05, .4])
    
    ax = fig.add_subplot(gs[0,2])
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['tessels'], method=['WTA', 'ridge', 'lasso'], summary_name=['weighted_all'])
    vis.plot_eval_predictions(dataframe=dataframe, plot_noiseceiling=True, normalize=False, hue='method', ax=ax)
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_xticks([80, 304, 670, 1190, 1848])

    ax = fig.add_subplot(gs[1,0])
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['tessels'], method=['WTA', 'ridge', 'lasso'], summary_name=['weighted_all'])
    vis.plot_eval_predictions(dataframe=dataframe, plot_noiseceiling=False, normalize=True, hue='method', ax=ax)
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_xticks([80, 304, 670, 1190, 1848])
    
    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_ridge_best_model_normalize.png')
    best_model = 'ridge_tessels1002_alpha_8'
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", model_name=best_model, normalize=True, method='ridge', outpath=fpath); # cscale=[0, 0.4]
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    # ax = fig.add_subplot(gs[1,1])
    # fpath = os.path.join(dirs.figure, f'map_R_lasso_best_model.png')
    # best_model = 'lasso_tessels1002_alpha_-2'
    # if not os.path.isfile(fpath):
    #     vis.map_eval_cerebellum(data="R", model_name=best_model, method='lasso', cscale=[0, 0.4], outpath=fpath); # ax=ax
    # vis.plot_png(fpath, ax=ax)
    # ax.axis('off')
    # ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'map_noiseceiling_XY_R_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="noiseceiling_XY_R", normalize=False, model_name='best_model', method='ridge', outpath=fpath); # ax=ax
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'F', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig2.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig3():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(5, 6, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax = fig.add_subplot(gs[0,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=0, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[1], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);
    # plt.ylim([0.58, .83]);

    ax = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg1.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[1], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg2.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=1, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,4])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[2], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[0,5])
    fpath = os.path.join(dirs.figure, f'MDTB-reg2.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[2], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg3.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=2, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,1])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[3], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg3.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[3], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg4.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=3, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,4])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[4], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[1,5])
    fpath = os.path.join(dirs.figure, f'MDTB-reg4.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[4], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[2,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg5.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=4, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[2,1])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[5], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[2,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg5.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[5], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[2,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg6.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=5, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'F', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[2,4])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[6], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[2,5])
    fpath = os.path.join(dirs.figure, f'MDTB-reg6.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[6], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[3,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg7.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=6, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'G', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[3,1])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[7], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[3,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg7.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[7], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[3,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg8.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=7, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'H', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[3,4])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[8], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[3,5])
    fpath = os.path.join(dirs.figure, f'MDTB-reg8.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[8], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[4,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg9.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=8, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'I', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[4,1])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[9], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[4,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg9.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[9], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[4,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg10.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=9, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'J', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[4,4])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, regions=[10], cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);

    ax = fig.add_subplot(gs[4,5])
    fpath = os.path.join(dirs.figure, f'MDTB-reg10.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[10], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig3.svg')
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

    ax = fig.add_subplot(gs[:,0])
    fpath = os.path.join(dirs.figure, f'group_lasso_percent_nonzero_cerebellum.png')
    if not os.path.isfile(fpath):
        vis.map_lasso_cerebellum(model_name='lasso_tessels0362_alpha_-2', stat='percent', weights='nonzero', stats=True, outpath=fpath);
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1:])
    ax,_ = vis.plot_surfaces(x='reg_names', hue=None, cortex='tessels0362', method='lasso', regions=None, stats=False, ax=ax);
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,1:])
    ax,_ = vis.plot_dispersion(y='var_w', hue=None, cortex='tessels0042', atlas='MDTB10',regions=None, stats=True, ax=ax);
    plt.ylim([0.58, .83]);
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig4.svg')
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

    ax = fig.add_subplot(gs[0,0])
    vis.plot_test_predictions(ax=ax, hue='test_routine')
    ax.set_xticks([80, 304, 670, 1190, 1848])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1])

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
    ax = fig.add_subplot(gs[0,0])
    vis.plot_train_predictions(dataframe=dataframe, x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['schaefer'], methods=['ridge'], ax=ax)
    ax.set_xlabel('Hyperparameter')
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([.05, .4])

    ax = fig.add_subplot(gs[0,1])
    vis.plot_train_predictions(dataframe=dataframe.query('train_hyperparameter>-5'), x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['schaefer'], methods=['lasso'], ax=ax)
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_ylim([.05, .4])
    
    ax = fig.add_subplot(gs[0,2])
    dataframe = vis.eval_summary(exps=['sc2'])
    vis.plot_predictions(dataframe=dataframe, exps=['sc2'], methods=['WTA', 'ridge'], atlases=['schaefer'], hue='model',  ax=ax) # 'lasso'
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    
    ax = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'map_R_ridge_schaefer_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='schaefer', model_name='best_model', method='ridge',  cscale=[0, 0.4], outpath=fpath);
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_lasso_schaefer_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='schaefer', model_name='best_model', method='lasso',  cscale=[0, 0.4], outpath=fpath); # ax=ax
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'map_R_WTA_schaefer_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='schaefer', model_name='best_model', method='WTA',  cscale=[0, 0.4], outpath=fpath) # ax=ax
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos+.05, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

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

    ax = fig.add_subplot(gs[0,0])
    vis.plot_surfaces(y='percent', cortex_group='tessels', weights='nonzero', regions=['Region3', 'Region6', 'Region7', 'Region8', 'Region9', 'Region10'], hue='reg_names', method='lasso', ax=ax);
    ax.set_xticks([80, 304, 670, 1190, 1848])
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg3.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=2, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg6.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=5, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg7.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=6, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg8.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=7, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg9.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=8, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'F', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg10.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=9, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'G', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,3])
    vis.plot_dispersion(hue='hem', y='Variance', cortex_group='tessels', cortex='tessels1002', atlas='MDTB10', regions=[3,6,7,8,9,10], ax=ax)
    ax.text(x_pos, y_pos, 'H', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS3.{format}')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
