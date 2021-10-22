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
    dataframe=None,
    exps=['sc1'], 
    x='train_num_regions', 
    hue=None, 
    atlases=None,
    save=False, 
    title=False,
    ax=None,
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

    # filer out atlases
    if atlases is not None:
        dataframe = dataframe[dataframe['train_atlas'].isin(atlases)]

    # filter data
    if methods is not None:
        dataframe = dataframe[dataframe['train_model'].isin(methods)]

    if (best_models):
        # get best model for each method
        model_names, _ = get_best_models(dataframe=dataframe)
        df1 = dataframe[dataframe['train_name'].isin(model_names)]
    else:
        df1 = dataframe

    df1['train_num_regions'] = df1['train_num_regions'].astype(int)
    # R
    ax = sns.lineplot(x=x, y="train_R_cv", hue=hue, data=df1, legend=True)
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
    save=False,
    atlases=['tessels'],
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

    # filer out atlases
    if atlases is not None:
        dataframe = dataframe[dataframe['eval_atlas'].isin(atlases)]

    # filter out methods
    if methods is not None:
        dataframe = dataframe[dataframe['eval_model'].isin(methods)]

    if noiseceiling:
        # #plt.figure(figsize=(8,8))
        ax = sns.lineplot(x=x, y="R_eval", hue=hue, legend=True, data=dataframe)
        ax = sns.lineplot(x=x, y='eval_noiseceiling_Y', data=dataframe, color='k', ax=ax, ci=None, linewidth=4)
        ax.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)
        plt.xticks(rotation="45", ha="right")
        ax.lines[-1].set_linestyle("--")
        ax.set_xlabel("")
        ax.set_ylabel("R")
    else:
        # #plt.figure(figsize=(8,8))
        sns.factorplot(x=x, y="R_eval", hue=hue, data=dataframe, legend=False, size=4, aspect=2) # size=4, aspect=2, order=x_order, hue_order=hue_order,,
        plt.xticks(rotation="45", ha="right")
        plt.xlabel("")
        plt.ylabel("R")
        if hue:
            plt.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)

    if title:
        plt.title("Model Evaluation", fontsize=20)
    
    if save:
        dirs = const.Dirs()
        exp_fname = '_'.join(exps)
        meth_fname = '_'.join(methods)
        plt.savefig(os.path.join(dirs.figure, f'eval_predictions_{exp_fname}_{meth_fname}_{x}.png'), pad_inches=0, bbox_inches='tight')

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

    # #plt.figure(figsize=(8, 8))
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

def plot_distances(
    exp='sc1',
    cortex='tessels1002',
    threshold=5,
    regions=[1,2,5],
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
        df = df[df['labels'].astype(int).isin(regions)]

    # filter out methods
    if cortex is not None:
        df = df[df['cortex'].isin([cortex])]
    
    # filter out methods
    if metric is not None:
        df = df[df['metric'].isin([metric])]

    # filter out methods
    if threshold is not None:
        df = df[df['threshold'].isin([threshold])]

    ax = sns.lineplot(x='labels', 
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
        plt.savefig(os.path.join(dirs.figure, f'cortical_distances_{exp}_{cortex}_{threshold}.png'), pad_inches=0, bbox_inches='tight')
    
    return df

def plot_surfaces(
            exp='sc1',
            y='percent',    
            cortex='tessels',
            weights='nonzero', 
            method='lasso',
            hue=None,
            regions=None,
            save=False 
            ):

    dirs = const.Dirs(exp_name=exp)
    
    # load in distances
    dataframe = pd.read_csv(os.path.join(dirs.conn_train_dir, 'cortical_surface_voxels_stats.csv')) 

    dataframe['num_regions'] = dataframe['cortex'].str.split('_').str.get(-1).str.extract('(\d+)').astype(float)
    dataframe['atlas'] = dataframe['cortex'].apply(lambda x: _add_atlas(x))

        # filter out methods
    if regions is not None:
        dataframe = dataframe[dataframe['labels'].astype(int).isin(regions)]

    # filter out methods
    if cortex is not None:
        dataframe = dataframe[dataframe['atlas'].isin([cortex])]

    # filter out methods
    if weights is not None:
        dataframe = dataframe[dataframe['weights'].isin([weights])]

    # filter out methods
    if method is not None:
        dataframe = dataframe[dataframe['method'].isin([method])]

    ax = sns.lineplot(x='num_regions', 
                y=y, 
                hue=hue, 
                data=dataframe,
                )
    ax.set_xlabel('')
    ax.set_ylabel('Percentage of cortical surface')
    plt.xticks(rotation="45", ha="right")
    if hue:
        plt.legend(loc='best', frameon=False) # bbox_to_anchor=(1, 1)
    
    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'cortical_surfaces_{exp}_{y}.png'), pad_inches=0, bbox_inches='tight')

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
    title=None):

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
        model_name, cortex = get_best_model(train_exp='sc1', method=method)
    
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

def subtract_AP_distances(
    model_name='ridge_tessels1002_alpha_8',
    threshold=100,
    method='ridge',
    atlas='MDTB10-subregions'
    ):
    
    dirs = const.Dirs(exp_name='sc1')
    
    if model_name=="best_model":
        model_name, cortex = get_best_model(train_exp='sc1', method=method)
    else:
        cortex = model_name.split('_')[1] # assumes model_name follows format: `<method>_<cortex>_alpha_<num>`
    
    # get model fpath
    fpath = os.path.join(dirs.conn_train_dir, model_name)
    
    # which regions are we subtracting?
    regions_to_subtract = {'L': [1, 11], 'R': [0,10]}
    
    giftis_all = []
    for k,v in regions_to_subtract.items():
        fname = f'group_{method}_{cortex}_{atlas}_threshold_{threshold}.{k}.func.gii'
        fpath = os.path.join(dirs.conn_train_dir, model_name, fname)
        data = load_surf_data(fpath)
        AP_subtract = data[:,v[0]] - data[:,v[1]]
        gifti = nio.make_func_gifti_cortex(data=AP_subtract, anatomical_struct=k)
        giftis_all.append(gifti)

    nio.view_cortex_inflated(giftis_all, column=None, borders=None, outpath=None)

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
        model,_ = get_best_model(train_exp=exp, method=method, atlas=atlas)

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
        model,_ = get_best_model(train_exp=exp, method='lasso', atlas=atlas)

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

def get_best_model(
    dataframe=None,
    train_exp='sc1',
    method=None,
    atlas=None
    ):
    """Get idx for best model based on either R_cv (or R_train)
    Args:
        dataframe (pd dataframe or None):
        train_exp (str): 'sc1' or 'sc2' or None (if dataframe is given)
        method (str or None): filter models by method
        atlas (str or None):
    Returns:
        model name (str)
    """

    # load train summary (contains R CV of all trained models)
    if dataframe is None:
        dataframe = train_summary(exps=[train_exp])

     # filter dataframe by method
    if method is not None:
        dataframe = dataframe[dataframe['train_model']==method]

    # filter dataframe by atlas
    if atlas is not None:
        dataframe = dataframe[dataframe['train_atlas']==atlas]

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
    method=None,
    roi=None
    ):
    """Get model_names, cortex_names for best models (NNLS, ridge, WTA) based on R_cv
    Args:
        dataframe (pd dataframe or None):
        train_exp (str): 'sc1' or 'sc2' or None (if dataframe is given)
        method (str or None): filter models by method
        roi (str or None): filter models by roi
    Returns:
        model_names (list of str), cortex_names (list of str)
    """
    # load train summary (contains R CV of all trained models)
    if dataframe is None:
        dataframe = train_summary(exps=[train_exp])

     # filter dataframe by method
    if method is not None:
        dataframe = dataframe[dataframe['train_model']==method]
    
    # filter dataframe by roi
    if roi is not None:
        dataframe = dataframe[dataframe['train_X_data']==roi]

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

