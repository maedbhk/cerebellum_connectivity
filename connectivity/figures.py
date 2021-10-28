
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
    vis.plot_eval_predictions(dataframe=dataframe, exps=['sc2'], atlases=['tessels'], methods=['WTA'], hue=None, save=True)
    ax3.text(x_pos, y_pos+.05, 'B', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    ax3.set_xticks([80, 304, 670, 1190, 1848])

    ax4 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_WTA_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='tessels', model_name='best_model', method='WTA', outpath=fpath) # ax=ax4
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
    vis.plot_eval_predictions(dataframe=dataframe, exps=['sc2'], methods=['WTA', 'ridge', 'lasso'], hue='eval_model', ax=ax3)
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    ax3.set_xticks([80, 304, 670, 1190, 1848])
    
    ax4 = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'map_R_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", model_name='best_model', method='ridge', outpath=fpath,  cscale=[0, 0.5]);
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    ax5 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_lasso_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", model_name='best_model', method='lasso', outpath=fpath, cscale=[0, 0.5]); # ax=ax4
    vis.plot_png(fpath, ax=ax5)
    ax5.axis('off')
    ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'map_noiseceiling_Y_R_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="noiseceiling_Y_R", exp="sc1", model_name='best_model', method='ridge', outpath=fpath); # ax=ax4
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
    gs = GridSpec(2, 4, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax1 = fig.add_subplot(gs[0,0])
    fpath = os.path.join(dirs.figure, f'group_lasso_percent_nonzero_cerebellum.png')
    if not os.path.isfile(fpath):
        vis.map_lasso_cerebellum(model_name='lasso_tessels1002_alpha_-2', exp="sc1", stat='percent', weights='nonzero', outpath=fpath);
    vis.plot_png(fpath, ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')

    ax2 = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'MDTB_motor_action_cognitive_regions.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=None)
    vis.plot_png(fpath, ax=ax2)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0,2])
    vis.plot_surfaces(y='percent', cortex='tessels', weights='nonzero', regions=['Region1', 'Region2', 'Region4', 'Region5'], hue='reg_names', method='lasso', ax=ax3);
    ax3.set_xticks([80, 304, 670, 1190, 1848])
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')

    ax4 = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=0, outpath=fpath)
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    ax5 = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg2.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=1, outpath=fpath)
    vis.plot_png(fpath, ax=ax5)
    ax5.axis('off')
    ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg4.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=3, outpath=fpath)
    vis.plot_png(fpath, ax=ax6)
    ax6.axis('off')
    ax6.text(x_pos, y_pos, 'F', transform=ax6.transAxes, fontsize=labelsize, verticalalignment='top')

    ax7 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg5.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=4, outpath=fpath)
    vis.plot_png(fpath, ax=ax7)
    ax7.axis('off')
    ax7.text(x_pos, y_pos, 'G', transform=ax7.transAxes, fontsize=labelsize, verticalalignment='top')

    # ax7 = fig.add_subplot(gs[1,2])
    # vis.plot_distances(exp='sc1', cortex='tessels1002', threshold=5, regions=['1', '2', '4', '5'], hue='hem', ax=ax7);
    # ax7.text(x_pos, y_pos, 'G', transform=ax7.transAxes, fontsize=labelsize, verticalalignment='top')

    ax8 = fig.add_subplot(gs[1,3])
    vis.plot_dispersion(hue='hem', y='Variance', cortex='tessels1002', regions=[1,2,4,5], ax=ax8)
    ax8.text(x_pos, y_pos, 'H', transform=ax8.transAxes, fontsize=labelsize, verticalalignment='top')
    # ax8.set_xticks([80, 304, 670, 1190, 1848])

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig3.{format}')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig4(format='png'):
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
    vis.plot_train_predictions(dataframe=dataframe, x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['mdtb'], methods=['ridge'], ax=ax1)
    ax1.set_xlabel('Hyperparameter')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')
    ax1.set_ylim([.05, .4])

    ax2 = fig.add_subplot(gs[0,1])
    # vis.plot_train_predictions(dataframe=dataframe.query('train_hyperparameter>-5'), x='train_hyperparameter', hue='train_num_regions', best_models=False, atlases=['mdtb'], methods=['lasso'], ax=ax2)
    # ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')
    # ax2.set_ylim([.05, .4])
    
    ax3 = fig.add_subplot(gs[0,2])
    dataframe = vis.eval_summary(exps=['sc2'])
    vis.plot_eval_predictions(dataframe=dataframe, exps=['sc2'], methods=['WTA', 'ridge'], atlases=['mdtb'], hue='eval_model',  ax=ax3) # 'lasso'
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=labelsize, verticalalignment='top')
    
    ax4 = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'map_R_mdtb_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='mdtb', model_name='best_model', method='ridge',  cscale=[0, 0.5], outpath=fpath);
    vis.plot_png(fpath, ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=labelsize, verticalalignment='top')

    ax5 = fig.add_subplot(gs[1,1])
    # fpath = os.path.join(dirs.figure, f'map_R_mdtb_lasso_best_model.png')
    # if not os.path.isfile(fpath):
    #     vis.map_eval_cerebellum(data="R", exp="sc1", atlas='mdtb', model_name='best_model', method='lasso',  cscale=[0, 0.5], outpath=fpath); # ax=ax4
    # vis.plot_png(fpath, ax=ax5)
    # ax5.axis('off')
    # ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=labelsize, verticalalignment='top')

    ax6 = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'map_R_WTA_mdtb_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", exp="sc1", atlas='mdtb', model_name='best_model', method='WTA',  cscale=[0, 0.5], outpath=fpath) # ax=ax4
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
    vis.plot_surfaces(y='percent', cortex='tessels', weights='nonzero', regions=['Region3', 'Region6', 'Region7', 'Region8', 'Region9', 'Region10'], hue='reg_names', method='lasso', ax=ax2);
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
    vis.plot_dispersion(hue='hem', y='Variance', cortex='tessels1002', regions=[3,6,7,8,9,10], ax=ax9)
    ax9.text(x_pos, y_pos, 'H', transform=ax9.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS3.{format}')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)


