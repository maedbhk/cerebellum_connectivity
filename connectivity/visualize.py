import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
from pathlib import Path
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

import connectivity.data as cdata
import connectivity.constants as const
import connectivity.nib_utils as nio

def plotting_style():
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 2
    plt.rc('xtick', labelsize=14)   
    plt.rc('ytick', labelsize=14)
    
    plt.rcParams.update({'font.size': 8})
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

    # look at model summary for train results
    df_concat = pd.DataFrame()
    for exp in exps:
        dirs = const.Dirs(exp_name=exp)
        fpath = os.path.join(dirs.conn_train_dir, f"{summary_name}.csv")
        df = pd.read_csv(fpath)
        df_concat = pd.concat([df_concat, df])

    # rename cols
    cols = []
    for col in df_concat.columns:
        if "train" not in col:
            cols.append("train_" + col)
        else:
            cols.append(col)

    df_concat.columns = cols

    if models_to_exclude:
        df_concat = df_concat[~df_concat['train_model'].isin(models_to_exclude)]

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

    # look at model summary for eval results
    df_concat = pd.DataFrame()
    for exp in exps:
        dirs = const.Dirs(exp_name=exp)
        fpath = os.path.join(dirs.conn_eval_dir, f"{summary_name}.csv")
        df = pd.read_csv(fpath)
        df_concat = pd.concat([df_concat, df])

    cols = []
    for col in df_concat.columns:
        if any(s in col for s in ("eval", "train")):
            cols.append(col)
        else:
            cols.append("eval_" + col)

    df_concat.columns = cols
    df_concat['eval_model'] = df_concat['eval_name'].str.split('_').str[0]

    # get noise ceilings
    df_concat["eval_noiseceiling_Y"] = np.sqrt(df_concat.eval_noise_Y_R)
    df_concat["eval_noiseceiling_XY"] = np.sqrt(df_concat.eval_noise_Y_R) * np.sqrt(df_concat.eval_noise_X_R)

    if models_to_exclude:
        df_concat = df_concat[~df_concat['eval_model'].isin(models_to_exclude)]

    return df_concat

def plot_train_predictions( 
    dataframe=None,
    exps=['sc1'], 
    x='train_name', 
    hue=None, 
    x_order=None, 
    hue_order=None, 
    save=True, 
    best_models=True,
    methods=['L2regression', 'WTA']
    ):
    """plots training predictions (R CV) for all models in dataframe.
    Args:
        exps (list of str): default is ['sc1']
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    if not dataframe:
        # get train summary
        dataframe = train_summary(exps=exps)

    # filter data
    dataframe = dataframe[dataframe['train_model'].isin(methods)]

    if (best_models):
        # get best model for each method
        df = dataframe.groupby(['train_X_data', 'train_model', 'train_hyperparameter']).mean().reset_index()

        df1 = df.groupby(['train_X_data', 'train_model']
                ).apply(lambda x: x['train_R_cv'].max()
                ).reset_index(name='train_R_cv')
    else:
        df1 = dataframe
    # R
    sns.factorplot(x=x, y="train_R_cv", hue=hue, data=df1, order=x_order, hue_order=hue_order, legend=False, ci=None, size=4, aspect=2)
    plt.title("Model Training (CV Predictions)", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xticks(rotation="45", ha="right")
    plt.xlabel("")
    plt.ylabel("R", fontsize=20)
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.show()

    if save:
        dirs = const.Dirs()
        exp_fname = '_'.join(exps)
        plt.savefig(os.path.join(dirs.figure, f'train_predictions_{exp_fname}.png'))

def plot_eval_predictions(
    dataframe=None,
    exps=['sc2'], 
    x='eval_X_data', 
    hue=None, 
    x_order=None, 
    hue_order=None, 
    save=True,
    methods=['ridge', 'WTA'],
    noiseceiling=True,
    ax=None,
    title=False
    ):
    """plots training predictions (R CV) for all models in dataframe.
    Args:
        exps (list of str): default is ['sc2']
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    if not dataframe:
        dataframe = eval_summary(exps=exps)

    # filter out methods
    dataframe = dataframe[dataframe['eval_model'].isin(methods)]

    # melt data into one column for easy plotting
    ax = sns.lineplot(x=x, y="R_eval", hue=hue, data=dataframe, legend=True, ax=ax) # size=4, aspect=2, order=x_order, hue_order=hue_order,
    plt.xticks(rotation="45", ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("R")

    if noiseceiling:
        ax = sns.lineplot(x=x, y='eval_noiseceiling_Y', data=dataframe, color='k', ax=ax)
        ax.lines[-1].set_linestyle("--")
        # sns.lineplot(x=x, y='eval_noiseceiling_XY', data=dataframe, color='k')
        ax.set_xlabel("")

    if title:
        plt.title("Model Evaluation", fontsize=20)
    
    if save:
        dirs = const.Dirs()
        exp_fname = '_'.join(exps)
        plt.savefig(os.path.join(dirs.figure, f'eval_predictions_{exp_fname}.png'))

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
    if not dataframe:
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
    colorbar=False, 
    cscale=None, 
    rois=True, 
    atlas='MDTB_10Regions', 
    save=True
    ):
    """plot surface map for best model
    Args:
        gifti (str):
        model (None or model name):
        exp (str): 'sc1' or 'sc2'
    """
    if exp == "sc1":
        dirs = const.Dirs(exp_name="sc2")
    else:
        dirs = const.Dirs(exp_name="sc1")

    # get best model
    model = model_name
    if model_name=="best_model":
        model,_ = get_best_model(train_exp=exp)

    # plot map
    fpath = os.path.join(dirs.conn_eval_dir, model)
    view = nio.view_cerebellum(gifti=os.path.join(fpath, f"group_{data}_vox.func.gii"), cscale=cscale, colorbar=colorbar)

    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'map_{data}_eval.png'))

    if rois:
        roi_summary(fpath=os.path.join(fpath, f"group_{data}_{model}_{exp}_vox.nii"), atlas=atlas)
    return view

def roi_summary(
    fpath, 
    atlas='MDTB_10Regions', 
    save=True
    ):
    """plot roi summary of data in `fpath`

    Args: 
        fpath (str): full path to nifti image
        atlas (str): default is 'MDTB_10Regions.nii'. assumes that atlases are saved in /cerebellar_atlases/
    Returns: 
        plots barplot of roi summary
    """
    dirs = const.Dirs()

    # get rois for `atlas`
    atlas_dir = os.path.join(dirs.base_dir, 'cerebellar_atlases')
    rois = cdata.read_suit_nii(atlas_dir + f'/{atlas}.nii')

    # get roi colors
    rgba, cpal = nio.get_gifti_colors(fpath=atlas_dir + f'/{atlas}.label.gii')
    labels = nio.get_gifti_labels(fpath=atlas_dir + f'/{atlas}.label.gii')

    df_all = pd.DataFrame()
    data = cdata.read_suit_nii(fpath)
    roi_mean, regs = cdata.average_by_roi(data, rois)
    fname = Path(fpath).stem
    df1 = pd.DataFrame({'roi_mean': list(np.hstack(roi_mean)),
                    'regions': list(regs),
                    'labels': list(labels),
                    'fnames': np.repeat(fname, len(regs))})
    
    plt.figure()
    sns.barplot(x='labels', y='roi_mean', data=df1.query('regions!=0'), palette=cpal[1:])
    plt.xticks(rotation=45)
    plt.xlabel(atlas)
    plt.ylabel('ROI mean')

    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'{atlas}_summary.png'))

    return df1

def map_model_comparison(
    model_name, 
    exp, 
    method='subtract', 
    colorbar=True, 
    rois=True, 
    atlas='MDTB_10Regions', 
    save=True
    ):
    """plot surface map for best model
    Args:
    """

    # initialize directories
    dirs = const.Dirs(exp_name=exp)

    fpath = os.path.join(dirs.conn_eval_dir, 'model_comparison')
    fpath_gii = glob.glob(f'{fpath}/*{method}*{model_name}*.gii*')
    fpath_nii = glob.glob(f'{fpath}/*{method}*{model_name}*.nii*')

    view = nio.view_cerebellum(fpath_gii[0], cscale=None, colorbar=colorbar)

    if save:
        dirs = const.Dirs()
        plt.savefig(os.path.join(dirs.figure, f'{model_name}_model_comparison_{exp}_{method}.png'))

    if rois:
        roi_summary(fpath=fpath_nii[0], atlas=atlas)
    
    return view

def map_weights(
    structure='cerebellum', 
    exp='sc1', 
    model_name='best_model', 
    hemisphere='R', 
    colorbar=False, 
    cscale=None, 
    rois=True, 
    atlas='MDTB_10Regions', 
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

    if (rois) & (structure=='cerebellum'):
        roi_summary(fpath=os.path.join(fpath, f"group_weights_cerebellum.nii"), atlas=atlas)
    return view

def map_atlas(
    fpath, 
    structure='cerebellum', 
    colorbar=False,
    title=False,
    outpath=None
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
        view = nio.view_cerebellum(gifti=fpath, colorbar=colorbar, outpath=outpath, title=title) 
    elif structure=='cortex':
        view = nio.view_cortex(gifti=fpath, outpath=outpath, title=title)
    else:
        print('please provide a valid parcellation')
    
    return view

def get_best_model(
    train_exp
    ):
    """Get idx for best model based on either R_cv (or R_train)
    Args:
        exp (str): 'sc1' or 'sc2
    Returns:
        model name (str)
    """
    # load train summary (contains R CV of all trained models)
    df = train_summary(exps=[train_exp])

    # get mean values for each model
    tmp = df.groupby(["name", "X_data"]).mean().reset_index()

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
    train_exp
    ):
    """Get model_names, cortex_names for best models (NNLS, ridge, WTA) based on R_cv
    Args:
        exp (str): 'sc1' or 'sc2
    Returns:
        model_names (list of str), cortex_names (list of str)
    """
    # load train summary (contains R CV of all trained models)
    df = train_summary(exps=[train_exp])

    tmp = df.groupby(['train_X_data', 'train_model', 'train_hyperparameter', 'train_name']
                ).mean().reset_index()

    tmp1 = tmp.groupby(['train_X_data', 'train_model']
            ).apply(lambda x: x['train_R_cv'].max()
            ).reset_index(name='train_R_cv')

    tmp2 = tmp1.merge(tmp, on=['train_X_data', 'train_model', 'train_R_cv'])

    model_names = list(tmp2['train_name'])

    cortex_names = list(tmp2['train_X_data'])

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

