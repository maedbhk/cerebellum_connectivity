import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
from nilearn.surface import load_surf_data
import re
from random import seed, sample

import connectivity.data as cdata
import connectivity.constants as const
import connectivity.nib_utils as nio

def plotting_style():
    plt.style.use('seaborn-poster') # ggplot
    params = {'axes.labelsize': 25,
            'axes.titlesize': 25,
            'legend.fontsize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            # 'figure.figsize': (10,5),
            'font.weight': 'regular',
            # 'font.size': 'regular',
            'font.family': 'sans-serif',
            'font.serif': 'Helvetica Neue',
            'lines.linewidth': 3,
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False}
    plt.rcParams.update(params)

def get_summary(
    summary_type= 'eval',
    summary_name=[None],
    exps=['sc2'],
    splitby=None,
    method=None,
    atlas=None
    ):
    """Appends different summary csv files (train or eval) and filters based on inputs
    Args:
        summary_type (str): 'eval','train'
        summary_name (list of str): name of summary file
        exps (list of str): default is ['sc2']
        splitby (list of str): splits to include is None
        method (list of str): methods to include
        atlas (list of str): atlasses to include
    Returns:
        pandas dataframe containing concatenated exp summary
    """

    # Get the names and exps into list of same length 
    if type(summary_name) is not list:
        summary_name=[summary_name]
    if type(exps) is not list:
        exps=[exps]*len(summary_name)

    # Load and concatenate the desired summary files 
    df_concat = pd.DataFrame()
    for exp,name in zip(exps,summary_name):
        dirs = const.Dirs(exp_name=exp)
        if name:
            fname = f"{summary_type}_summary_{name}.csv"
        else:
            fname = f"{summary_type}_summary.csv"
        if summary_type=="eval":
            fpath = os.path.join(dirs.conn_eval_dir, fname)
        elif summary_type =="train":
            fpath = os.path.join(dirs.conn_train_dir, fname)
        df = pd.read_csv(fpath)
        df_concat = pd.concat([df_concat, df])

    # add atlas and method
    df_concat['atlas'] = df_concat['X_data'].apply(lambda x: _add_atlas(x))
    df_concat['method'] = df_concat['name'].str.split('_').str.get(0)

    # Training specific items: 
    if summary_type == 'train':
        df_concat['hyperparameter'] = df_concat['hyperparameter'].astype(float) 

    # Evaluation specific items: 
    if summary_type=='eval':
        df_concat['noiseceiling_Y']=np.sqrt(df_concat.noise_Y_R)
        df_concat['noiseceiling_XY']=np.sqrt(df_concat.noise_Y_R * df_concat.noise_X_R)

    # Now filter the data frame
    if splitby is not None:
        df_concat = df_concat[df_concat['splitby'].isin(splitby)]
    if method is not None:
        df_concat = df_concat[df_concat['method'].isin(method)]
    if atlas is not None:
        df_concat = df_concat[df_concat['atlas'].isin(atlas)]

    return df_concat

def test_summary(
    summary_name="test_summary_learning",
    train_exp='sc1',
    models_to_include=['RIDGE']
    ):
    """
    JD:I GUESS THIS IS FOR GENERALIZATION TESTING.
    YOU LIKELY DO NOT NEED THIS - INSTEAD YOU SHOULD BE ABLE TO
    RELY ON
    load test summary containing all metrics about test models (mdtb models tested on external data).
    Prefix 'gen' is appended to cols.
    Args:
        summary_name (str): name of summary file
        train_exp (str): default is 'sc1'
        models_to_include (list of str):
    Returns:
        pandas dataframe containing test summary
    """
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.data_dir, "conn_models", f"{summary_name}.csv")
    df = pd.read_csv(fpath)

    df['atlas'] = df['X_data'].apply(lambda x: _add_atlas(x))

    cols = []
    for col in df.columns:
        if any(s in col for s in ("eval", "train")):
            cols.append(col)
        else:
            cols.append("test_" + col)

    df.columns = cols
    df['test_model'] = df['test_name'].str.split('_').str[0]

    # get noise ceilings
    df["test_noiseceiling_Y"] = np.sqrt(df.test_noise_Y_R)
    # df["test_noiseceiling_XY"] = np.sqrt(df.test_noise_Y_R) * np.sqrt(df.test_noise_X_R)

    if models_to_include:
        df = df[df['test_model'].isin(models_to_include)]

    return df

def roi_summary(
    data_fpath,
    atlas_nifti,
    atlas_gifti,
    plot=False
    ):
    """plot roi summary of data in `fpath`

    Args:
        data_fpath (str): full path to nifti image
        atlas_nifti (str): full path to nifti atlas  (e.g., ./MDTB_10Regions.nii')
        atlas_gifti (str): full path to gifti atlas (e.g., ./MDTB_10Regions.label.gii)
        plot (bool): default is False
    Returns:
        dataframe (pd dataframe)
    """
    # get rois for `atlas`
    rois = cdata.read_suit_nii(atlas_nifti)

    # get roi colors
    rgba, cpal, cmap = nio.get_gifti_colors(fpath=atlas_gifti, ignore_0=True)
    labels = nio.get_gifti_labels(fpath=atlas_gifti)

    data = cdata.read_suit_nii(data_fpath)
    roi_mean, regs = cdata.average_by_roi(data, rois)
    df = pd.DataFrame({'mean': list(np.hstack(roi_mean)),
                    'regions': list(regs),
                    'labels': list(labels)
                    })
    if plot:
        #plt.figure(figsize=(8,8))
        df = df.query('regions!=0')
        sns.barplot(x='labels', y='mean', data=df, palette=cpal)
        plt.xticks(rotation='45')
        plt.xlabel('')
    return df

def _add_atlas(x):
    """returns abbrev. atlas name from `X_data` column
    """
    atlas = x.split('_')[0]
    atlas = ''.join(re.findall(r"[a-zA-Z]", atlas)).lower()

    return atlas

def plot_train_predictions(
    dataframe,
    x='num_regions',
    hue=None,
    save=False,
    title=False,
    ax=None):
    """plots training predictions (R CV) for all models in dataframe.
    Args:
        dataframe: Training data frame from get_summary
        exps (list of str): default is ['sc1']
        hue (str or None): can be 'exp', 'Y_data' etc.
    """
    ax = sns.lineplot(x=x, y="R_cv", hue=hue, data=dataframe, legend=True)
    ax.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)
    plt.xticks(rotation="45", ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("R (CV)")

    if hue is not None:
        plt.legend(loc='best', frameon=False)
    plt.xticks(rotation="45", ha="right")
    # ax.lines[-1].set_linestyle("--")
    plt.xlabel("")
    plt.ylabel("R (cv)")
    if title:
        plt.title("Model Training (CV Predictions)", fontsize=20)

    if save:
        dirs = const.Dirs()
        fname = f'train_predictions_{x}'
        if hue:
            fname = f'train_predictions_{hue}_{x}'
        plt.savefig(os.path.join(dirs.figure, f'{fname}.svg'), pad_inches=0.1, bbox_inches='tight')

def plot_eval_predictions(
    dataframe,
    x='num_regions',
    hue=None,
    save=False,
    ax=None,
    title=False,
    ):
    """plots eval predictions (R CV) for all models in dataframe.
    Args:
        exps (list of str): default is ['sc2']
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    ax = sns.lineplot(x=x, y="R_eval", hue=hue, data=dataframe) # legend=True,
    ax = sns.lineplot(x=x, y='noiseceiling_Y', data=dataframe, color='k', ax=ax, ci=None, linewidth=4)
    ax.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)
    plt.xticks(rotation="45", ha="right")
    ax.lines[-1].set_linestyle("--")
    ax.set_xlabel("")
    ax.set_ylabel("R")

    if title:
        plt.title("Model Evaluation", fontsize=20)

    if save:
        dirs = const.Dirs()
        fname = f'eval_predictions_{x}.svg'
        if hue:
            fname = f'eval_predictions_{hue}_{x}.svg'
        plt.savefig(os.path.join(dirs.figure, fname, pad_inches=0, bbox_inches='tight'))

def plot_test_predictions(
    dataframe,
    x='num_regions',
    hue=None,
    save=False,
    noiseceiling='group',
    ax=None,
    title=False,
    ):
    """plots eval predictions (R CV) for all models in dataframe.
    Args:
        exps (list of str): default is ['sc2']
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    ax = sns.lineplot(x=x, y="R_eval", hue=hue, ci=70, color='g', legend=True, data=dataframe, ax=ax)
    ax = sns.lineplot(x=x, y='test_noiseceiling_Y', data=dataframe, color='k', ax=ax, ci=None, linewidth=4)
    ax.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)
    plt.xticks(rotation="45", ha="right")
    ax.lines[-1].set_linestyle("--")
    # ax.lines[0].set_color('g')
    ax.set_xlabel("")
    ax.set_ylabel("R")

    if title:
        plt.title("Model Generalization", fontsize=20)

    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'test_predictions_learning.png'), pad_inches=0, bbox_inches='tight')

    return dataframe

def plot_distances(
    exp='sc1',
    cortex='tessels1002',
    threshold=5,
    regions=['1', '2', '5'], # '01A', '02A'
    hue='hem',
    metric='gmean',
    title=False,
    save=False,
    ax=None):

    dirs = const.Dirs(exp_name=exp)

    # load in distances
    df = pd.read_csv(os.path.join(dirs.conn_train_dir, 'cortical_distances_stats.csv'))

    df['threshold'] = df['threshold']*100
    df['labels'] = df['labels'].str.replace(re.compile('Region|-'), '', regex=True)
    df['subregion'] = df['labels'].str.replace(re.compile('[^a-zA-Z]'), '', regex=True)
    df['num_regions'] = df['cortex'].str.split('_').str.get(-1).str.extract('(\d+)')

    # filter out methods
    if regions is not None:
        df = df[df['labels'].isin(regions)]

    # filter out methods
    if cortex is not None:
        df = df[df['cortex'].isin([cortex])]

    # filter out methods
    if metric is not None:
        df = df[df['metric'].isin([metric])]

    # filter out methods
    if threshold is not None:
        df = df[df['threshold'].isin([threshold])]

    ax = sns.boxplot(x='labels',
                y='distance',
                hue=hue,
                data=df,
                )
    ax.set_xlabel('Cerebellar Regions')
    ax.set_ylabel('Average Cortical Distance')
    plt.xticks(rotation="45", ha="right")
    if hue:
        plt.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)

    if title:
        plt.title("Cortical Distances", fontsize=20)

    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'cortical_distances_{exp}_{cortex}_{threshold}.svg'), pad_inches=0, bbox_inches='tight')

    return df

def plot_surfaces(
    exp='sc1',
    x='reg_names',
    y='percent',    
    cortex_group='tessels',
    cortex='tessels1002',
    weights='nonzero', 
    method='lasso',
    atlas='MDTB10',
    hue=None,
    regions=None,
    save=False,
    ax=None,
    palette=None
    ):

    dirs = const.Dirs(exp_name=exp)

    # load in distances
    dataframe_vox = pd.read_csv(os.path.join(dirs.conn_train_dir, 'cortical_surface_voxels_stats.csv')) 
    dataframe_roi = pd.read_csv(os.path.join(dirs.conn_train_dir, f'cortical_surface_rois_stats_{atlas}.csv')) 
    dataframe_concat = pd.concat([dataframe_vox, dataframe_roi]) 

    # dataframe['subregion'] = dataframe['reg_names'].str.replace(re.compile('[^a-zA-Z]'), '', regex=True)
    dataframe_concat['num_regions'] = dataframe_concat['cortex'].str.split('_').str.get(-1).str.extract('(\d+)').astype(float)*2
    dataframe_concat['cortex_group'] = dataframe_concat['cortex'].apply(lambda x: _add_atlas(x))

    # filter 
    if regions is not None:
        dataframe_concat = dataframe_concat[dataframe_concat['reg_names'].isin(regions)]
    if cortex_group is not None:
        dataframe_concat = dataframe_concat[dataframe_concat['cortex_group'].isin([cortex_group])]
    if cortex is not None:
        dataframe_concat = dataframe_concat[dataframe_concat['cortex'].isin([cortex])]
    if atlas is not None:
        dataframe_concat = dataframe_concat[dataframe_concat['atlas'].isin([atlas])]
    if weights is not None:
        dataframe_concat = dataframe_concat[dataframe_concat['weights'].isin([weights])]
    if method is not None:
        dataframe_concat = dataframe_concat[dataframe_concat['method'].isin([method])]

    # color plot according to MDTB10 atlas
    fpath = nio.get_cerebellar_atlases(atlas_keys=['atl-MDTB10'])[0]
    _, cpal, _ = nio.get_gifti_colors(fpath)
    palette = cpal

    if x=='num_regions':
        ax = sns.lineplot(x=x, 
                    y=y, 
                    hue=hue, 
                    data=dataframe_concat,
                    palette=palette,
                    )
    else:
        ax = sns.barplot(x=x, 
            y=y, 
            hue=hue, 
            data=dataframe_concat,
            palette=palette,
            )
    ax.set_xlabel('')
    ax.set_ylabel('Percentage of cortical surface')
    plt.xticks(rotation="45", ha="right")
    if hue:
        plt.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)

    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'cortical_surfaces_{exp}_{y}.svg'), pad_inches=0, bbox_inches='tight')

    return dataframe_concat

def plot_dispersion(
    exp='sc1',
    y='Variance',    
    cortex='tessels1002', 
    cortex_group='tessels',
    atlas='MDTB10',
    method='ridge',
    hue='hem',
    regions=None, # [1,2,5]
    save=False,
    ax=None
    ):

    dirs = const.Dirs(exp_name=exp)

    # load in distances
    dataframe = pd.read_csv(os.path.join(dirs.conn_train_dir, f'cortical_dispersion_stats_{atlas}.csv'))

    dataframe['w_var']=dataframe.Variance*dataframe.sum_w
    dataframe['var_w'] = dataframe.w_var/dataframe.sum_w
    dataframe['hem'] = dataframe['hem'].map({0: 'L', 1: 'R'})
    dataframe['num_regions'] = dataframe['cortex'].str.split('_').str.get(-1).str.extract('(\d+)').astype(float)*2
    dataframe['cortex_group'] = dataframe['cortex'].apply(lambda x: _add_atlas(x))

    # filter
    if cortex is not None:
        dataframe = dataframe[dataframe['cortex'].isin([cortex])]
    if cortex_group is not None:
        dataframe = dataframe[dataframe['cortex_group'].isin([cortex_group])]
    if atlas is not None:
        dataframe = dataframe[dataframe['atlas'].isin([atlas])]
    if method is not None:
        dataframe = dataframe[dataframe['method'].isin([method])]
    if regions is not None:
        dataframe = dataframe[dataframe['roi'].isin(regions)]

    # T = pd.pivot_table(dataframe,values=['sum_w','w_var','Variance'],index=['subj','roi', 'hem'],aggfunc='mean')
    # T = T.reset_index()
    # T['var_w'] = T.w_var/T.sum_w
    ax = sns.boxplot(x='roi', 
                y=y, 
                hue=hue, 
                data=dataframe,
                )
    ax.set_xlabel('')
    ax.set_ylabel('Cortical Dispersion')
    plt.xticks(rotation="45", ha="right")
    if hue:
        plt.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)

    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'cortical_dispersion_{y}.svg'), pad_inches=0, bbox_inches='tight')

    return dataframe

def map_distances_cortex(
    atlas='MDTB10',
    threshold=1,
    column=0,
    borders=False,
    model_name='best_model',
    method='ridge',
    surf='flat',
    colorbar=True,
    outpath=None,
    title=None
    ):

    """plot cortical map for distances
    Args:
        atlas (str): default is 'MDTB10'
        threshold (int): default is 1
        column (int): default is 0
        exp (str): 'sc1' or 'sc2'
        borders (bool): default is False
        model_name (str): default is 'best_model'
        method (str): 'ridge' or 'lasso'
        hemisphere (str): 'L' or 'R'
        colorbar (bool): default is True
        outpath (str or None): default is None
        title (bool): default is True
    """
    dirs = const.Dirs(exp_name='sc1')

    # get best model
    if model_name=="best_model":
        dataframe = get_summary(exps=['sc1'], summary_type='train', method=[method])
        model_name, cortex = get_best_model(dataframe)

    giftis = []
    for hemisphere in ['L', 'R']:
        fname = f'group_{atlas}_threshold_{threshold}.{hemisphere}.func.gii'
        fpath = os.path.join(dirs.conn_train_dir, model_name, fname)
        giftis.append(fpath)

    for i, hem in enumerate(['L', 'R']):
        if surf=='flat':
            nio.view_cortex(giftis[i], surf=surf, hemisphere=hem, title=title, column=column, colorbar=colorbar, outpath=outpath)

    if surf=='inflated':
        nio.view_cortex_inflated(giftis, column=column, borders=borders, outpath=outpath)

def map_eval_cerebellum(
    data="R",
    exp="sc1",
    model_name='best_model',
    method='ridge',
    atlas='tessels',
    colorbar=True,
    cscale=None,
    outpath=None,
    title=None,
    new_figure=True
    ):
    """plot surface map for best model
    Args:
        data (str): 'R', 'R2', 'noiseceiling_Y_R' etc.
        exp (str): 'sc1' or 'sc2'
        model_name ('best_model' or model name):
    """
    if exp == "sc1":
        dirs = const.Dirs(exp_name="sc2")
    else:
        dirs = const.Dirs(exp_name="sc1")

    # get best model
    model = model_name
    if model_name=="best_model":
        dataframe = get_summary(exps=['sc1'], summary_type='train', method=[method], atlas=[atlas])
        model_name, cortex = get_best_model(dataframe)

    fpath = os.path.join(dirs.conn_eval_dir, model, f'group_{data}_vox.func.gii')
    view = nio.view_cerebellum(gifti=fpath, cscale=cscale, colorbar=colorbar,
                    new_figure=new_figure, title=title, outpath=outpath);

    return view

def map_lasso_cerebellum(
    model_name,
    exp="sc1",
    stat='percent',
    weights='nonzero',
    atlas='tessels',
    method='lasso',
    colorbar=False,
    cscale=None,
    outpath=None,
    title=None,
    new_figure=True
    ):
    """plot surface map for best model
    Args:
        model (None or model name):
        exp (str): 'sc1' or 'sc2'
        stat (str): 'percent' or 'count'
    """
    dirs = const.Dirs(exp_name=exp)

    # get best model
    model = model_name
    if model_name=="best_model":
        dataframe = get_summary(exps=['sc1'], summary_type='train', method=[method], atlas=[atlas])
        model_name, cortex = get_best_model(dataframe)

    # plot map
    fpath = os.path.join(dirs.conn_train_dir, model)

    fname = f"group_lasso_{stat}_{weights}_cerebellum"
    gifti = os.path.join(fpath, f'{fname}.func.gii')
    view = nio.view_cerebellum(gifti=gifti,
                            cscale=cscale,
                            colorbar=colorbar,
                            title=title,
                            outpath=outpath,
                            new_figure=new_figure
                            )
    return view

def map_weights(
    structure='cerebellum',
    exp='sc1',
    model_name='best_model',
    method='ridge',
    atlas='tessels',
    hemisphere='R',
    colorbar=False,
    cscale=None,
    save=True
    ):
    """plot training weights for cortex or cerebellum
    Args:
        gifti_func (str): '
    """
    # initialise directories
    dirs = const.Dirs(exp_name=exp)

    # get best model
    model = model_name
    if model_name=='best_model':
        dataframe = get_summary(exps=['sc1'], summary_type='train', method=[method], atlas=[atlas])
        model_name, cortex = get_best_model(dataframe)

    # get path to model
    fpath = os.path.join(dirs.conn_train_dir, model)

    outpath = None
    if save:
        dirs = const.Dirs()
        outpath = os.path.join(dirs.figure, f'{model}_{structure}_{hemisphere}_weights_{exp}.png')

    # plot either cerebellum or cortex
    if structure=='cerebellum':
        surf_fname = fpath + f'/group_weights_{structure}.func.gii'
        view = nio.view_cerebellum(gifti=surf_fname, cscale=cscale, colorbar=colorbar, outpath=outpath)
    elif structure=='cortex':
        surf_fname =  fpath + f"/group_weights_{structure}.{hemisphere}.func.gii"
        view = nio.view_cortex(gifti=surf_fname, cscale=cscale, outpath=outpath)
    else:
        print("gifti must contain either cerebellum or cortex in name")

    return view

def get_best_model(dataframe):
    """Get idx for best model based on either R_cv (or R_train)
    Args:
        dataframe (pd dataframe ):
            Data frame with training summary data (from get_summary)
    Returns:
        model name (str)
    """

    # get mean values for each model
    tmp = dataframe.groupby(["name", "X_data"]).mean().reset_index()

    # get best model (based on R CV or R train)
    try:
        best_model = tmp[tmp["R_cv"] == tmp["R_cv"].max()]["name"].values[0]
        cortex = tmp[tmp["R_cv"] == tmp["R_cv"].max()]["X_data"].values[0]
    except:
        best_model = tmp[tmp["R_train"] == tmp["R_train"].max()]["name"].values[0]
        cortex = tmp[tmp["R_train"] == tmp["R_train"].max()]["X_data"].values[0]

    print(f"best model is {best_model}")

    return best_model, cortex

def get_best_models(dataframe):
    """Get model_names, cortex_names for best models (NNLS, ridge, WTA) based on R_cv from train_summary
    Args:
        dataframe (pd dataframe ):
    Returns:
        model_names (list of str), cortex_names (list of str)
    """
    df_mean = dataframe.groupby(['X_data', 'method', 'name'], sort=True).apply(lambda x: x['R_cv'].mean()).reset_index(name='R_cv_mean')
    df_best = df_mean.groupby(['X_data', 'method']).apply(lambda x: x[['name', 'R_cv_mean']].max()).reset_index()

    tmp = dataframe.groupby(['X_data', 'method', 'hyperparameter', 'name']).mean().reset_index()

    # group by `X_data` and `model`
    grouped =  tmp.groupby(['X_data', 'method'])

    model_names = []; cortex_names = []
    for name, group in grouped:
        model_name = group.sort_values(by='R_cv', ascending=False)['name'].head(1).tolist()[0]
        cortex_name = group.sort_values(by='R_cv', ascending=False)['X_data'].head(1).tolist()[0]
        model_names.append(model_name)
        cortex_names.append(cortex_name)

    return model_names, cortex_names

def plot_distance_matrix(
    roi='tessels0042',
    ):
    """Plot matrix of distances for cortical `roi`
    Args:
        roi (str): default is 'tessels0042'
    Returns:
        plots distance matrix
    """

    # get distances for `roi` and `hemisphere`
    distances = cdata.get_distance_matrix(roi=roi)[0]

    # visualize matrix of distances
    ##plt.figure(figsize=(8,8))
    plt.imshow(distances)
    plt.colorbar()
    plt.show()

    return distances

def plot_png(
    fpath,
    ax=None
    ):
    """ Plots a png image from `fpath`

    Args:
        fpath (str): full path to image to plot
        ax (bool): figure axes. Default is None
    """
    if os.path.isfile(fpath):
        img = mpimg.imread(fpath)
    else:
        print("image does not exist")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.imshow(img, origin='upper', vmax=abs(img).max(), vmin=-abs(img).max(), aspect='equal')

def join_png(
    fpaths,
    outpath=None,
    offset=0
    ):
    """Join together pngs into one png

    Args:
        fpaths (list of str): full path(s) to images
        outpath (str): full path to new image. If None, saved in current directory.
        offset (int): default is 0.
    """

    # join images together
    images = [Image.open(x) for x in fpaths]

    # resize all images (keep ratio aspect) based on size of min image
    sizes = ([(np.sum(i.size), i.size ) for i in images])
    min_sum = sorted(sizes)[0][0]

    images_resized = []
    for s, i in zip(sizes, images):
        resize_ratio = int(np.floor(s[0] / min_sum))
        orig_size = list(s[1])
        if resize_ratio>1:
            resize_ratio = resize_ratio - 1.5
        new_size = tuple([int(np.round(x / resize_ratio)) for x in orig_size])
        images_resized.append(Image.fromarray(np.asarray(i.resize(new_size))))

    widths, heights = zip(*(i.size for i in images_resized))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for im in images_resized:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0] - offset

    if not outpath:
        outpath = 'concat_image.png'
    new_im.save(outpath)

    return new_im

