
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import numpy as np
import os
import pandas as pd
from scipy import stats as sp
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
    x_pos = -0.1
    y_pos = 1.1
    labelsize = 40

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0,0])
    ax = fig2_plot_eval_uncorrected(save=False, ax=ax)
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    
    ax = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'map_noiseceiling_Y_R_ridge_best_model.png')
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="noiseceiling_Y_R", normalize=False, model_name='best_model', method='ridge', outpath=fpath); # ax=ax
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,1])
    ax = fig2_plot_eval_corrected(save=False, ax=ax)
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    
    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'map_R_ridge_best_model_normalize.png')
    best_model = 'ridge_tessels1002_alpha_8'
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="R", model_name=best_model, normalize=True, method='ridge', outpath=fpath); # cscale=[0, 0.4]
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig2.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig2_plot_eval_uncorrected(save=False, ax=None):

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 40

    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['tessels'], method=['WTA', 'ridge', 'lasso'], summary_name=['weighted_all'])
    df, ax = vis.plot_eval_predictions(dataframe=dataframe, noiseceiling='Y', normalize=False, hue='method', ax=ax, save=save)
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_xticks([80, 304, 670, 1190, 1848])
    ax.set_xticklabels([80, 304, 670, 1190, 1848], rotation=45)

    # RIDGE AND LASSO STAT: T TEST
    result = sp.ttest_rel(df.ridge, df.lasso, nan_policy='omit')
    res = [f'T stat {t:.3f}: pval {r:.3f}' for (t,r) in zip(result.statistic, result.pvalue)]
    print(f'T test for evaluation between ridge and lasso for TESSELS is: {res}')
    
    #  RIDGE AND LASSO STAT: F TEST - MODELS X GRANULARITY
    result = sp.f_oneway(df.ridge.mean(), df.lasso.mean())
    print(f'F test for ridge and lasso is {result}')

    #  RIDGE AND WTA STAT: T TEST
    result = sp.ttest_rel(df.ridge, df.WTA, nan_policy='omit')
    res = [f'T stat {t:.3f}: pval {r:.3f}' for (t,r) in zip(result.statistic, result.pvalue)]
    print(f'T test for evaluation between ridge and WTA for TESSELS is: {res}')
    
    #  RIDGE AND WTA STAT: F TEST - MODELS X GRANULARITY
    result = sp.f_oneway(df.ridge.mean(), df.WTA.mean())
    print(f'F test for ridge and WTA is {result}')

    # mean predictive accuracy of lasso + ridge models
    mean_accuracy = (df.ridge.mean().mean() + df.lasso.mean().mean()) / 2
    print(f'mean predictive accuracy for lasso + ridge is {mean_accuracy}')

    # mean predictive accuracy of WTA model
    mean_accuracy = df.WTA.mean().mean()
    print(f'mean predictive accuracy for WTA is {mean_accuracy}')

    # cerebellar reliability
    mean, std = dataframe.groupby('subj_id')['noiseceiling_Y'].mean().agg({'mean', 'std'}) 
    print(f'mean cerebellar reliability is {mean} (std: {std})')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    return ax

def fig2_plot_eval_corrected(save=False, ax=None):

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 40

    # YEO
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['yeo'], cortex=['yeo7'], method=['WTA', 'ridge', 'lasso'], summary_name=['weighted_all'])
    df, ax = vis.plot_eval_predictions(dataframe=dataframe, noiseceiling=None, normalize=True, plot_type='point', hue='method', ax = ax, save = save)

    #  RIDGE AND WTA STAT: T TEST
    result = sp.ttest_rel(df.ridge, df.WTA, nan_policy='omit')
    res = [f'T stat {t:.3f}: pval {r:.3f}' for (t,r) in zip(result.statistic, result.pvalue)]
    print(f'T test for evaluation between ridge and WTA for YEO is: {res}')
    
    #  RIDGE AND LASSO STAT: T TEST
    result = sp.ttest_rel(df.ridge, df.lasso, nan_policy='omit')
    res = [f'T stat {t:.3f}: pval {r:.3f}' for (t,r) in zip(result.statistic, result.pvalue)]
    print(f'T test for evaluation between ridge and lasso for YEO is: {res}')

    # TESSELS
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['tessels'], method=['WTA', 'ridge', 'lasso'], summary_name=['weighted_all'])
    df, _ = vis.plot_eval_predictions(dataframe=dataframe, noiseceiling=None, normalize=True, hue='method', ax=ax)
    ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.set_xticks([7, 80, 304, 670, 1190, 1848])
    ax.set_xticklabels([7, 80, 304, 670, 1190, 1848], rotation=45)

    # RIDGE AND LASSO STAT: T TEST
    result = sp.ttest_rel(df.ridge, df.lasso, nan_policy='omit')
    res = [f'T stat {t:.3f}: pval {r:.3f}' for (t,r) in zip(result.statistic, result.pvalue)]
    print(f'T test for evaluation between ridge and lasso for TESSELS is: {res}')
    
    #  RIDGE AND LASSO STAT: F TEST - MODELS X GRANULARITY
    result = sp.f_oneway(df.ridge.mean(), df.lasso.mean())
    print(f'F test for ridge and lasso for TESSELS is {result}')

    #  RIDGE AND WTA STAT: T TEST
    result = sp.ttest_rel(df.ridge, df.WTA, nan_policy='omit')
    res = [f'T stat {t:.3f}: pval {r:.3f}' for (t,r) in zip(result.statistic, result.pvalue)]
    print(f'T test for evaluation between ridge and WTA for TESSELS is: {res}')
    
    #  RIDGE AND WTA STAT: F TEST - MODELS X GRANULARITY
    result = sp.f_oneway(df.ridge.mean(), df.WTA.mean())
    print(f'F test for ridge and WTA for TESSELS is {result}')

    # mean predictive accuracy of lasso + ridge models
    mean_accuracy = (df.ridge.mean().mean() + df.lasso.mean().mean()) / 2
    print(f'mean predictive accuracy for lasso + ridge for TESSELS is {mean_accuracy}')

    # mean predictive accuracy of WTA model
    mean_accuracy = df.WTA.mean().mean()
    print(f'mean predictive accuracy for WTA for TESSELS is {mean_accuracy}')

    # cerebellar reliability
    mean, std = dataframe.groupby('subj_id')['noiseceiling_Y'].mean().agg({'mean', 'std'}) 
    print(f'mean cerebellar reliability is {mean} (std: {std})')
    
    return ax

def fig3():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(6, 4, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=0, outpath=fpath, colorbar=True)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,0])
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

    ax = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg2.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[2], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg3.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=2, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0])
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

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg4.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[4], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[2,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg5.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=4, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[2,0])
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

    ax = fig.add_subplot(gs[2,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg6.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[6], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[3,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg7.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=6, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'G', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[3,0])
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

    ax = fig.add_subplot(gs[3,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg8.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[8], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[4,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg9.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', surf='inflated', threshold=100, column=8, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'I', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[4,0])
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

    ax = fig.add_subplot(gs[4,2])
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
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 50

    ax = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_lasso_percent_nonzero_cerebellum.png')
    if not os.path.isfile(fpath):
        vis.map_surface_cerebellum(model_name='lasso_tessels0042_alpha_-3', stat='percent', colorbar=True, weights='nonzero', outpath=fpath);
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,0])
    ax,df = vis.plot_surfaces(x='regions', hue=None, cortex='tessels0042', method='lasso', voxels=True, regions=None, ax=ax);
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    plt.ylabel('% of cortical surface', fontsize=35)
    # plt.ylim([0.2, 2.2])
    # plt.yticks([0.2, 0.6, 1.0, 1.4, 1.8, 2.2], fontsize=40)
    plt.xticks(fontsize=40)
    result = sp.f_oneway(df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], df[10])
    print(f'F stat for surfaces: {result.statistic:.3f}: pval {result.pvalue:.3f}')

    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_lasso_dispersion_cerebellum.png')
    if not os.path.isfile(fpath):
        vis.map_dispersion_cerebellum(model_name='lasso_tessels0042_alpha_-3', method='lasso', colorbar=True, stat='var_w', atlas='tessels', outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0])
    ax,df = vis.plot_dispersion(y='var_w', hue=None, y_label='cortical dispersion', cortex='tessels0042', method='lasso', voxels=True, atlas='MDTB10', regions=None, ax=ax);
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    plt.ylabel('dispersion', fontsize=35)
    # plt.ylim([0.2, 0.6])
    # plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6], fontsize=40)
    plt.xticks(fontsize=40)
    result = sp.f_oneway(df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], df[10])
    print(f'F stat dispersion {result.statistic:.3f}: pval {result.pvalue:.3f}')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'fig4.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def fig5():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax1 = fig.add_subplot(gs[0,0])
    dataframe = vis.get_summary_learning(summary_name='learning', 
                        atlas=['icosahedron'], 
                        method=['WTA', 'ridge', 'lasso'],
                        routine=['session_1', 'session_2', 'session_3'],
                        incl_rest=True,
                        incl_instruct=False,
                        task_subset=None
                        )
    vis.plot_eval_predictions(dataframe, x='num_regions', normalize=True, noiseceiling=None, hue='method', ax=ax1)
    ax1.set_xticks([80, 304, 670, 1190, 1848])
    ax1.text(x_pos, y_pos, '', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, f'fig5.svg'), bbox_inches="tight", dpi=300)

def figS1():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 40

    ax = fig.add_subplot(gs[0,0])
    
    # RIDGE
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['shen', 'gordon', 'fan', 'arslan', 'schaefer', 'yeo'], method=['ridge'], summary_name=['weighted_all'])
    df,ax = vis.plot_eval_predictions(dataframe=dataframe, noiseceiling=None, plot_type='point', normalize=True, hue='atlas', markers=["p", "x", "s", ".", ">", "^"], palette=None, color=(0,0,0), linestyles=['-', '-', '-', '-', '-' , '-'], ax=ax) 
    
    # WTA
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['shen', 'gordon', 'fan', 'arslan', 'schaefer', 'yeo'], method=['WTA'], summary_name=['weighted_all'])
    df,ax = vis.plot_eval_predictions(dataframe=dataframe, noiseceiling=None, plot_type='point', normalize=True, hue='atlas', markers=["p", "x", "s", ".", ">", "^"], palette=None, color=(0,0,0), linestyles=[':', ':', ':', ':', ':', ':'], ax=ax) 
    ax.set_ylim([0.4, 0.7])
    # ax.set_xticks([7, 80, 304, 670, 1190, 1848])
    # ax.set_xticklabels([7, 80, 304, 670, 1190, 1848], rotation=45);
    
    # do statistics
    dataframe = vis.get_summary('eval', exps=['sc2'], atlas=['shen', 'gordon', 'fan', 'arslan', 'schaefer', 'yeo'], method=['ridge', 'WTA'], summary_name=['weighted_all'])
    df,_ = vis.plot_eval_predictions(dataframe=dataframe, plot_type=None) 
    result = sp.ttest_rel(df.ridge, df.WTA, nan_policy='omit')
    print(f'F test for evaluation between WTA and ridge for FUNCTIONAL is: {result}')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS1.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS2():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(3, 4, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax = fig.add_subplot(gs[0,0])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg1A.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[1], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1A.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=0, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg1B.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[11], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg2A.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=10, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg2A.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[2], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1B.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=1, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg2B.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[12], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg2B.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=11, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[2,0:2])
    ax,df = vis.plot_surfaces(x='regions', hue=None, cortex='tessels0362', atlas='MDTB10-subregions', method='lasso', regions=[1,2,11,12], ax=ax);
    ax.text(x_pos, y_pos, 'L', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    plt.ylim([0.1, 0.5])
    plt.ylabel('% of cortical surface', fontsize=35)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], fontsize=40)
    plt.xticks(fontsize=40)
    result = sp.f_oneway(df[1], df[2], df[11], df[12])
    print(f'F test for surfaces is {result}')

    ax = fig.add_subplot(gs[2,2:])
    ax,df = vis.plot_dispersion(y='var_w', hue=None, y_label='cortical dispersion', cortex='tessels0042', method='ridge', atlas='MDTB10-subregions', regions=[1,2,11,12], ax=ax);
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    plt.ylim([0.65, 0.85])
    plt.ylabel('dispersion', fontsize=35)
    plt.yticks([0.65, 0.75, 0.85], fontsize=40)
    plt.xticks(fontsize=40)
    result = sp.f_oneway(df[1], df[2], df[11], df[12])
    print(f'F test for dispersion is {result}')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS2.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS3():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(6, 4, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg1.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=0, outpath=fpath, colorbar=True)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,0])
    fpath = os.path.join(dirs.figure, f'MDTB-reg1.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[1], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg2.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=1, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')

    ax = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg2.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[2], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg3.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=2, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'MDTB-reg3.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[3], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,3])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg4.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=3, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg4.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[4], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[2,1])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg5.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=4, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'E', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[2,0])
    fpath = os.path.join(dirs.figure, f'MDTB-reg5.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[5], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[2,3])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg6.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=5, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'F', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[2,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg6.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[6], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[3,1])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg7.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=6, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'G', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[3,0])
    fpath = os.path.join(dirs.figure, f'MDTB-reg7.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[7], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[3,3])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg8.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=7, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'H', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[3,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg8.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[8], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[4,1])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg9.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=8, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'I', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[4,0])
    fpath = os.path.join(dirs.figure, f'MDTB-reg9.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[9], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[4,3])
    fpath = os.path.join(dirs.figure, f'group_distances_lasso_best_model_MDTB10-reg10.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10', method='lasso', surf='inflated', threshold=100, column=9, outpath=fpath, colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'J', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[4,2])
    fpath = os.path.join(dirs.figure, f'MDTB-reg10.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[10], atlas='MDTB10_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS3.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS4():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 50

    ax = fig.add_subplot(gs[0,0])
    ax,df = vis.plot_surfaces(x='regions', hue=None, cortex='tessels0042', method='ridge', regions=None, ax=ax);
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    plt.ylabel('% of cortical surface', fontsize=35)
    # plt.ylim([2, 12])
    # plt.yticks([2, 4, 6, 8, 10, 12], fontsize=40)
    plt.xticks(fontsize=40)
    result = sp.f_oneway(df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], df[10])
    print(f'F stat surfaces {result.statistic:.3f}: pval {result.pvalue:.3f}')

    ax = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_ridge_percent_nonzero_cerebellum.png')
    if not os.path.isfile(fpath):
        vis.map_surface_cerebellum(model_name='ridge_tessels0042_alpha_4', method='ridge', stat='percent', colorbar=True, weights='nonzero', outpath=fpath);
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'B', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0])
    ax,df = vis.plot_dispersion(y='var_w', hue=None, y_label='cortical dispersion', cortex='tessels0042', method='ridge', atlas='MDTB10', regions=None, ax=ax);
    ax.text(x_pos, y_pos, 'C', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    plt.ylabel('Dispersion', fontsize=35)
    # plt.ylim([0.65, 0.85])
    # plt.yticks([0.65, 0.7, 0.75, 0.8, 0.85], fontsize=40)
    plt.xticks(fontsize=40)
    result = sp.f_oneway(df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], df[10])
    print(f'F stat dispersion {result.statistic:.3f}: pval {result.pvalue:.3f}')

    ax = fig.add_subplot(gs[1,1])
    ax.text(x_pos, y_pos, 'D', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    fpath = os.path.join(dirs.figure, f'group_ridge_dispersion_cerebellum.png')
    if not os.path.isfile(fpath):
        vis.map_dispersion_cerebellum(model_name='ridge_tessels0042_alpha_4', colorbar=True, stat='var_w', atlas='tessels', outpath=fpath)

    vis.plot_png(fpath, ax=ax)
    ax.axis('off')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    save_path = os.path.join(dirs.figure, f'figS4.svg')
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def figS5():

    dirs = const.Dirs()
    vis.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 50

    ax1 = fig.add_subplot(gs[0,0])
    df = vis.get_summary("train", exps='sc1', method=['ridge'], atlas=['tessels'], summary_name=[''])
    vis.plot_train_predictions(df, x='hyperparameter', hue='num_regions', ax=ax1)
    ax1.set_xlabel('Hyperparameter')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=labelsize, verticalalignment='top')
    ax1.set_ylim([.15, .35])
    print(df.groupby(['num_regions', 'hyperparameter'])['R_cv'].agg({'mean', 'std'}))

    ax2 = fig.add_subplot(gs[0,1])
    df = vis.get_summary("train", exps='sc1', method=['lasso'], atlas=['tessels'], summary_name=[''])
    vis.plot_train_predictions(df, x='hyperparameter', hue='num_regions', ax=ax2)
    ax2.set_xlabel('Hyperparameter')
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=labelsize, verticalalignment='top')
    ax2.set_ylim([.15, .35])
    plt.xticks([15, 20, 25])
    print(df.groupby(['num_regions', 'hyperparameter'])['R_cv'].agg({'mean', 'std'}))

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, f'figS5.svg'), bbox_inches="tight", dpi=300)

def figS6():
    dirs = const.Dirs()
    vis.plotting_style()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 50

    ax = fig.add_subplot(gs[0,0])
    fpath = os.path.join(dirs.figure, f'map_noiseceiling_cerebellum.png')
    best_model = 'ridge_tessels1002_alpha_8'
    if not os.path.isfile(fpath):
        vis.map_eval_cerebellum(data="noiseceiling_Y_R", model_name=best_model, normalize=False, method='ridge', outpath=fpath); # cscale=[0, 0.4]
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, 'A', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, f'figS6.svg'), bbox_inches="tight", dpi=300)

def figS7():
    dirs = const.Dirs()
    vis.plotting_style()

    fig = plt.figure(figsize=(4,4))
    gs = GridSpec(1, 1, figure=fig)

    vis.plot_surfaces(x='num_regions', hue=None, cortex=None, voxels=True, average_regions=True, cortex_group='tessels', method='lasso', plot_type='line', regions=[1,2,7,8]);
    plt.ylabel('% of cortical surface', fontsize=35)
    plt.xticks([80, 304, 670, 1190, 1848])
    # plt.xticklabels([80, 304, 670, 1190, 1848])

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, f'figS7.svg'), bbox_inches="tight", dpi=300)

def figS8():
    plt.clf()
    vis.plotting_style()

    dirs = const.Dirs()

    fig = plt.figure()
    gs = GridSpec(2, 4, figure=fig)

    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax = fig.add_subplot(gs[0,0])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg1A.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[1], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[0,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1A.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=0, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[0,2])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg1B.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[11], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[0,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg2A.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=10, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,0])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg2A.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[2], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,1])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg1B.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=1, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    ax = fig.add_subplot(gs[1,2])
    fpath = os.path.join(dirs.figure, f'MDTB-subregions-reg2B.png')
    if not os.path.isfile(fpath):
        nio.view_atlas_cerebellum(outpath=fpath, labels=[12], atlas='MDTB10-subregions_dseg', colorbar=False)
    vis.plot_png(fpath, ax=ax)
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')
    ax.axis('off')

    ax = fig.add_subplot(gs[1,3])
    fpath = os.path.join(dirs.figure, f'group_distances_best_model_MDTB10-reg2B.png')
    if not os.path.isfile(fpath):
        vis.map_distances_cortex(model_name='best_model', atlas='MDTB10-subregions', surf='inflated', threshold=100, column=11, outpath=fpath)
    vis.plot_png(fpath, ax=ax)
    ax.axis('off')
    ax.text(x_pos, y_pos, '', transform=ax.transAxes, fontsize=labelsize, verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    plt.savefig(os.path.join(dirs.figure, f'figS8.svg'), bbox_inches="tight", dpi=300)