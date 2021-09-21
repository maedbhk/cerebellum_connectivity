import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
from pathlib import Path
import matplotlib.image as mpimg
from  matplotlib.ticker import FuncFormatter
from PIL import Image
import matplotlib.pyplot as plt
import re
from random import seed, sample

import connectivity.data as cdata
import connectivity.constants as const
import connectivity.nib_utils as nio

def plotting_style():
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 3
    plt.rc('xtick', labelsize=18)   
    plt.rc('ytick', labelsize=18)
    
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'svg'
    plt.rcParams["savefig.dpi"] = 300
    plt.rc("axes.spines", top=False, right=False) # removes certain axes
    plt.rcParams["axes.grid"] = False

def _concat_summary(summary_name='train_summary'):
    """concat dataframes from different experimenters

    Args: 
        summary_name (str): 'train_summary' or 'eval_summary'

    Returns: 
        saves concatenated dataframe <summary_name>.csv to disk
    """
    for exp in ['sc1', 'sc2']:

        dirs = const.Dirs(exp_name=exp)
        
        if summary_name=='train_summary':
            os.chdir(dirs.conn_train_dir)
        elif summary_name=='eval_summary':
            os.chdir(dirs.conn_eval_dir)
       
        files = glob.glob(f'*{summary_name}_*')

        df_all = pd.DataFrame()
        for file in files:
            df = pd.read_csv(file)
            df_all = pd.concat([df_all, df])

        df_all.to_csv(f'{summary_name}.csv')

def split_subjects(
    subj_ids, 
    test_size=0.3
    ):
    """Randomly divide subject list into train and test subsets.

    Train subjects are used to train, validate, and test models(s).
    Test subjects are kept until the end of the project to evaluate
    the best (and final) model.

    Args:
        subj_ids (list): list of subject ids (e.g., ['s01', 's02'])
        test_size (int): size of test set
    Returns:
        train_subjs (list of subject ids), test_subjs (list of subject ids)
    """
    # set random seed
    seed(1)

    # get number of subjects in test (round down)
    num_in_test = int(np.floor(test_size * len(subj_ids)))

    # select test set
    test_subjs = list(sample(subj_ids, num_in_test))
    train_subjs = list([x for x in subj_ids if x not in test_subjs])

    return train_subjs, test_subjs

def train_summary(
    summary_name="train_summary",
    exps=['sc1'], 
    models_to_exclude=['NNLS', 'PLSRegress']
    ):
    """load train summary containing all metrics about training models.
    Summary across exps is concatenated and prefix 'train' is appended to cols.
    Args:
        summary_name (str): name of summary file
        exps (list of str): default is ['sc1', 'sc2']
        models_to_exclude (list of str or None):
    Returns:
        pandas dataframe containing concatenated exp summary
    """
    # concat summary
    _concat_summary(summary_name)

    train_subjs, _ = split_subjects(const.return_subjs)

    # look at model summary for train results
    df_concat = pd.DataFrame()
    for exp in exps:
        dirs = const.Dirs(exp_name=exp)
        fpath = os.path.join(dirs.conn_train_dir, f"{summary_name}.csv")
        df = pd.read_csv(fpath)
        df_concat = pd.concat([df_concat, df])

    # select trained subjects 
    df_concat = df_concat[df_concat['subj_id'].isin(train_subjs)]
    
    df_concat['atlas'] = df_concat['X_data'].apply(lambda x: _add_atlas(x))

    # rename cols
    cols = []
    for col in df_concat.columns:
        if "train" not in col:
            cols.append("train_" + col)
        else:
            cols.append(col)

    df_concat.columns = cols

    try: 
        # add hyperparameter for wnta models (was NaN before) - this should be fixed in modeling routine
        # add NTakeAll number to `train_model` (WNTA_N<1>)
        wnta = df_concat.query('train_model=="WNTA"')
        wnta['train_hyperparameter'] = wnta['train_name'].str.split('_').str.get(-1)
        wnta['train_model'] = wnta['train_model'] + '_' + wnta['train_name'].str.split('_').str.get(-3)

        # get rest of dataframe
        other = df_concat.query('train_model!="WNTA"')

        #concat dataframes
        df_concat = pd.concat([wnta, other])
    except: 
        pass
    
    df_concat['train_hyperparameter'] = df_concat['train_hyperparameter'].astype(float) # was float

    if models_to_exclude:
        df_concat = df_concat[~df_concat['train_model'].isin(models_to_exclude)]

    def _relabel_model(x):
        if x=='L2regression':
            return 'ridge'
        elif x=='LASSO':
            return 'lasso'
        else:
            return x

    df_concat['train_model'] = df_concat['train_model'].apply(lambda x: _relabel_model(x))

    return df_concat

def eval_summary(
    summary_name="eval_summary",
    exps=['sc2'], 
    models_to_exclude=['NNLS', 'PLSRegress']
    ):
    """load eval summary containing all metrics about eval models.
    Summary across exps is concatenated and prefix 'eval' is appended to cols.
    Args:
        summary_name (str): name of summary file
        exps (list of str): default is ['sc1', 'sc2']
    Returns:
        pandas dataframe containing concatenated exp summary
    """
    # concat summary
    _concat_summary(summary_name)

    train_subjs, _ = split_subjects(const.return_subjs)

    # look at model summary for eval results
    df_concat = pd.DataFrame()
    for exp in exps:
        dirs = const.Dirs(exp_name=exp)
        fpath = os.path.join(dirs.conn_eval_dir, f"{summary_name}.csv")
        df = pd.read_csv(fpath)
        df_concat = pd.concat([df_concat, df])

    # select trained subjects 
    df_concat = df_concat[df_concat['subj_id'].isin(train_subjs)]
    
    df_concat['atlas'] = df_concat['X_data'].apply(lambda x: _add_atlas(x))

    cols = []
    for col in df_concat.columns:
        if any(s in col for s in ("eval", "train")):
            cols.append(col)
        else:
            cols.append("eval_" + col)

    df_concat.columns = cols
    df_concat['eval_model'] = df_concat['eval_name'].str.split('_').str[0]

    try: 
        wnta = df_concat.query('eval_model=="wnta"')
        wnta['eval_model'] = wnta['eval_model'] + '_' + wnta['eval_name'].str.split('_').str.get(-3)

        # get rest of dataframe
        other = df_concat.query('eval_model!="wnta"')

        #concat dataframes
        df_concat = pd.concat([wnta, other])
    except:
        pass

    # get noise ceilings
    df_concat["eval_noiseceiling_Y"] = np.sqrt(df_concat.eval_noise_Y_R)
    df_concat["eval_noiseceiling_XY"] = np.sqrt(df_concat.eval_noise_Y_R) * np.sqrt(df_concat.eval_noise_X_R)

    if models_to_exclude:
        df_concat = df_concat[~df_concat['eval_model'].isin(models_to_exclude)]

    return df_concat

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
        plt.figure(figsize=(8,8))
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
    dataframe=None,
    exps=['sc1'], 
    x='train_num_regions', 
    hue=None, 
    x_order=None, 
    hue_order=None, 
    save=True, 
    title=False,
    best_models=True,
    methods=['ridge', 'WTA', 'lasso']
    ):
    """plots training predictions (R CV) for all models in dataframe.
    Args:
        exps (list of str): default is ['sc1']
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    if dataframe is None:
        # get train summary
        dataframe = train_summary(exps=exps)

    # filter data
    dataframe = dataframe[dataframe['train_model'].isin(methods)]

    if (best_models):
        # get best model for each method
        model_names, _ = get_best_models(dataframe=dataframe)
        df1 = dataframe[dataframe['train_name'].isin(model_names)]
    else:
        df1 = dataframe
    # R
    # ax = sns.factorplot(x=x, y="train_R_cv", hue=hue, data=df1, order=x_order, hue_order=hue_order, legend=False, size=4, aspect=2)
    plt.figure(figsize=(8,8))
    ax = sns.lineplot(x=x, y="train_R_cv", hue=hue, data=df1)
    plt.xticks(rotation="45", ha="right")

    # ax = sns.lineplot(x=x, y="train_R_cv", hue=hue, data=df1)
    if hue is not None:
        plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.xticks(rotation="45", ha="right")
    # ax.lines[-1].set_linestyle("--")
    plt.xlabel("")
    plt.ylabel("R (cv)")
    if title:
        plt.title("Model Training (CV Predictions)", fontsize=20)

    if save:
        dirs = const.Dirs()
        exp_fname = '_'.join(exps)
        meth_fname = '_'.join(methods)
        if hue:
            fname = f'train_predictions_{exp_fname}_{meth_fname}_{hue}_{x}'
        else:
            fname = f'train_predictions_{exp_fname}_{meth_fname}_{x}'
        plt.savefig(os.path.join(dirs.figure, f'{fname}.png'), pad_inches=0.1, bbox_inches='tight')

def plot_eval_predictions(
    dataframe=None,
    exps=['sc2'], 
    x='eval_num_regions', 
    hue=None, 
    save=True,
    methods=['ridge', 'WTA'],
    noiseceiling=True,
    ax=None,
    title=False,
    ):
    """plots eval predictions (R CV) for all models in dataframe.
    Args:
        exps (list of str): default is ['sc2']
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    if dataframe is None:
        dataframe = eval_summary(exps=exps)

    # filter out methods
    dataframe = dataframe[dataframe['eval_model'].isin(methods)]

    if noiseceiling:
        plt.figure(figsize=(8,8))
        ax = sns.lineplot(x=x, y="R_eval", hue=hue, legend=True, data=dataframe)
        ax = sns.lineplot(x=x, y='eval_noiseceiling_Y', data=dataframe, color='k', ax=ax, ci=None, linewidth=4)
        ax.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        plt.xticks(rotation="45", ha="right")
        ax.lines[-1].set_linestyle("--")
        ax.set_xlabel("")
        ax.set_ylabel("R")
    else:
        plt.figure(figsize=(8,8))
        sns.factorplot(x=x, y="R_eval", hue=hue, data=dataframe, legend=False, size=4, aspect=2) # size=4, aspect=2, order=x_order, hue_order=hue_order,,
        plt.xticks(rotation="45", ha="right")
        plt.xlabel("")
        plt.ylabel("R")
        if hue:
            plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    if title:
        plt.title("Model Evaluation", fontsize=20)
    
    if save:
        dirs = const.Dirs()
        exp_fname = '_'.join(exps)
        meth_fname = '_'.join(methods)
        plt.savefig(os.path.join(dirs.figure, f'eval_predictions_{exp_fname}_{meth_fname}_{x}.png'), pad_inches=0, bbox_inches='tight')

def plot_predictions_atlas(
    data='eval', 
    method='WTA',
    save=True,
    format='png'
    ):
    """plot eval predictions
    """
    if data=='eval':
        df = eval_summary()
        df = df[df['eval_model'].isin([method])]
        x='eval_num_regions'; y="R_eval"; hue='eval_atlas'
    elif data=='train':
        df = train_summary()
        df = df[df['train_model'].isin([method])]
        x='train_num_regions'; y='train_R_cv'; hue='train_atlas'
                                                
    paper_rc = {'lines.linewidth': 3}                  
    sns.set_context("paper", rc=paper_rc) 
    sns.factorplot(x=x, y=y, hue=hue, legend=False, data=df, ax=None, size=4, aspect=2)
    plt.xticks(rotation='45', fontsize=18);
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('R', fontsize=20)
    plt.xlabel('# of regions', fontsize=20);
    paper_rc = {'lines.linewidth': 6}                  
    sns.set_context("paper", rc=paper_rc) 
    
    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'{method}_predictions_{data}_atlas.{format}'), format=format, dpi=300, bbox_inches="tight")

def plot_best_eval(
    dataframe=None,
    exps=['sc2'],
    save=True
    ):
    """plots evaluation predictions (R eval) for best model in dataframe for 'sc1' or 'sc2'
    Also plots model-dependent and model-independent noise ceilings.
    Args:
        exps (list of str): default is ['sc2']
        hue (str or None): default is 'eval_name'
    """
    if dataframe is None:
        # get evaluation dataframe
        dataframe = eval_summary(exps=exps)

    dataframe_all = pd.DataFrame()
    for exp in exps:

        if exp is "sc1":
            train_exp = "sc2"
        else:
            train_exp = "sc1"

        best_model,_ = get_best_model(train_exp=train_exp)

        dataframe = dataframe.query(f'eval_exp=="{exp}" and eval_name=="{best_model}"')
        dataframe_all = pd.concat([dataframe_all, dataframe])

    # melt data into one column for easy plotting
    cols = ["eval_noiseceiling_Y", "eval_noiseceiling_XY", "R_eval"]
    df = pd.melt(dataframe_all, value_vars=cols, id_vars=set(dataframe_all.columns) - set(cols)).rename(
        {"variable": "data_type", "value": "data"}, axis=1
    )

    plt.figure(figsize=(8, 8))
    splot = sns.barplot(x="data_type", y="data", data=df)
    plt.title(f"Model Evaluation: best model={best_model})", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel("")
    plt.ylabel("R", fontsize=20)
    plt.xticks(
        [0, 1, 2], ["noise ceiling (data)", "noise ceiling (model)", "model predictions"], rotation="45", ha="right"
    )
    # annotate barplot
    for p in splot.patches:
        splot.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="left",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )
    plt.show()
    if save:
        dirs = const.Dirs()
        exp_fname = '_'.join(exps)
        plt.savefig(os.path.join(dirs.figure, f'best_eval_{exp_fname}.png'))

def map_eval(
    data="R", 
    exp="sc1", 
    model_name='best_model', 
    method='ridge',
    colorbar=False, 
    cscale=None,  
    save=True,
    title=True
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
        model,_ = get_best_model(train_exp=exp, method=method)

    # plot map
    fname = f"group_{data}_vox.func.gii"
    view = nio.view_cerebellum(gifti=os.path.join(dirs.conn_eval_dir, model, fname), cscale=cscale, colorbar=colorbar, title=title, save=save)

    return view

def map_lasso(
    model_name,
    exp="sc1", 
    stat='percent',
    colorbar=False, 
    cscale=None,  
    save=True,
    title=None
    ):
    """plot surface map for best model
    Args:
        model (None or model name):
        exp (str): 'sc1' or 'sc2'
        stat (str): 'percent' or 'count'
    """
    dirs = const.Dirs(exp_name=exp)

    # plot map
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    fname = f"group_lasso_{stat}_positive_cerebellum"

    view = nio.view_cerebellum(gifti=os.path.join(fpath, f'{fname}.func.gii'), cscale=cscale, colorbar=colorbar, title=title, save=save)

    return view

def map_model_comparison(
    model_name, 
    exp, 
    colorbar=True, 
    save=True,
    title=True
    ):
    """plot surface map for best model
    Args:
    """

    # initialize directories
    dirs = const.Dirs(exp_name=exp)

    fpath = os.path.join(dirs.conn_eval_dir, 'model_comparison')
    fpath_gii = glob.glob(f'{fpath}/*subtract*{model_name}*.gii*')
    fpath_nii = glob.glob(f'{fpath}/*subtract*{model_name}*.nii*')

    view = nio.view_cerebellum(fpath_gii[0], cscale=None, colorbar=colorbar, title=title, save=save)
    
    return view

def map_weights(
    structure='cerebellum', 
    exp='sc1', 
    model_name='best_model', 
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
        model,_ = get_best_model(train_exp=exp)
    
    # get path to model
    fpath = os.path.join(dirs.conn_train_dir, model)

    # plot either cerebellum or cortex
    if structure=='cerebellum':
        surf_fname = fpath + f'/group_weights_{structure}.func.gii'
        view = nio.view_cerebellum(gifti=surf_fname, cscale=cscale, colorbar=colorbar)
    elif structure=='cortex':
        surf_fname =  fpath + f"/group_weights_{structure}.{hemisphere}.func.gii"
        view = nio.view_cortex(gifti=surf_fname, cscale=cscale)
    else:
        print("gifti must contain either cerebellum or cortex in name")
    
    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'{model}_{structure}_{hemisphere}_weights_{exp}.png'))

    return view

def map_atlas(
    fpath, 
    structure='cerebellum', 
    colorbar=False,
    title=False,
    save=True,
    ):
    """General purpose function for plotting (optionally saving) *.label.gii or *.func.gii parcellations (cortex or cerebellum)
    Args: 
        fpath (str): full path to atlas
        structure (str): default is 'cerebellum'. other options: 'cortex'
        colorbar (bool): default is False. If False, saves colorbar separately to disk.
        save (bool): default is True.
    Returns:
        viewing object to visualize parcellations
    """
    if structure=='cerebellum':
        view = nio.view_cerebellum(gifti=fpath, colorbar=colorbar, title=title, save=save) 
    elif structure=='cortex':
        view = nio.view_cortex(gifti=fpath, title=title, save=save)
    else:
        print('please provide a valid parcellation')
    
    return view

def get_best_model(
    dataframe=None,
    train_exp='sc1',
    method=None,
    ):
    """Get idx for best model based on either R_cv (or R_train)
    Args:
        dataframe (pd dataframe or None):
        train_exp (str): 'sc1' or 'sc2' or None (if dataframe is given)
        method (str or None): filter models by method
    Returns:
        model name (str)
    """

    # load train summary (contains R CV of all trained models)
    if dataframe is None:
        dataframe = train_summary(exps=[train_exp])

     # filter dataframe by method
    if method is not None:
        dataframe = dataframe[dataframe['train_model']==method]

    # get mean values for each model
    tmp = dataframe.groupby(["train_name", "train_X_data"]).mean().reset_index()

    # get best model (based on R CV or R train)
    try: 
        best_model = tmp[tmp["train_R_cv"] == tmp["train_R_cv"].max()]["train_name"].values[0]
        cortex = tmp[tmp["train_R_cv"] == tmp["train_R_cv"].max()]["train_X_data"].values[0]
    except:
        best_model = tmp[tmp["R_train"] == tmp["R_train"].max()]["train_name"].values[0]
        cortex = tmp[tmp["R_train"] == tmp["R_train"].max()]["train_X_data"].values[0]

    print(f"best model for {train_exp} is {best_model}")

    return best_model, cortex

def get_best_models(
    dataframe=None,
    train_exp='sc1',
    method=None
    ):
    """Get model_names, cortex_names for best models (NNLS, ridge, WTA) based on R_cv
    Args:
        dataframe (pd dataframe or None):
        train_exp (str): 'sc1' or 'sc2' or None (if dataframe is given)
        method (str or None): filter models by method
    Returns:
        model_names (list of str), cortex_names (list of str)
    """
    # load train summary (contains R CV of all trained models)
    if dataframe is None:
        dataframe = train_summary(exps=[train_exp])

     # filter dataframe by method
    if method is not None:
        dataframe = dataframe[dataframe['train_model']==method]

    df_mean = dataframe.groupby(['train_X_data', 'train_model', 'train_name'], sort=True).apply(lambda x: x['train_R_cv'].mean()).reset_index(name='train_R_cv_mean')
    df_best = df_mean.groupby(['train_X_data', 'train_model']).apply(lambda x: x[['train_name', 'train_R_cv_mean']].max()).reset_index()

    tmp = dataframe.groupby(['train_X_data', 'train_model', 'train_hyperparameter', 'train_name']
                ).mean().reset_index()

    # group by `X_data` and `model`
    grouped =  tmp.groupby(['train_X_data', 'train_model'])

    model_names = []; cortex_names = []
    for name, group in grouped:
        model_name = group.sort_values(by='train_R_cv', ascending=False)['train_name'].head(1).tolist()[0]
        cortex_name = group.sort_values(by='train_R_cv', ascending=False)['train_X_data'].head(1).tolist()[0]
        model_names.append(model_name)
        cortex_names.append(cortex_name)

    return model_names, cortex_names

def get_eval_models(
    exp
    ):
    df = eval_summary(exps=[exp])
    df = df[['eval_name', 'eval_X_data']].drop_duplicates() # get unique model names

    return df['eval_name'].to_list(), np.unique(df['eval_X_data'].to_list())

def show_distance_matrix(
    roi='tessels0042'
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
    plt.imshow(distances)
    plt.colorbar()
    plt.show()

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

