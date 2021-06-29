
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import numpy as np
import os
import pandas as pd

from connectivity import visualize
import connectivity.constants as const 

def plotting_style():
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 6
    plt.rc('xtick', labelsize=14)     
    plt.rc('ytick', labelsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'svg'
    plt.rc("axes.spines", top=False, right=False) # removes certain axes

def fig1():
    plt.clf()

    plotting_style()

    # fig = plt.figure()
    fig = plt.figure(figsize=(15,15))
    gs = GridSpec(3, 2, figure=fig)

    x_pos = -0.2
    y_pos = 1.02

    ax1 = fig.add_subplot(gs[0, 0])
    # visualize.png_plot(filename=atlas, ax=ax1)
    ax1.text(x_pos, 1.0, 'A', transform=ax1.transAxes, fontsize=60, verticalalignment='top')
    ax1.yaxis.label.set_size(50)
    ax1.tick_params(axis='x', which='major', labelsize=50, rotation=45)
    ax1.axis('off')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    
    dirs = const.Dirs()
    plt.savefig(os.path.join(dirs.figure, 'fig1'), bbox_inches="tight", dpi=300)
