import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
import glob
import deepdish as dd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

import connectivity.data as cdata
import connectivity.constants as const
import connectivity.nib_utils as nio

plt.rcParams["axes.grid"] = False

def train_summary(summary_name="train_summary"):
    """load train summary containing all metrics about training models.
    Summary across exps is concatenated and prefix 'train' is appended to cols.
    Args:
        summary_name (str): name of summary file
    Returns:
        pandas dataframe containing concatenated exp summary
    """
    # look at model summary for train results
    df_concat = pd.DataFrame()
    for exp in ["sc1", "sc2"]:
        dirs = const.Dirs(exp_name=exp)
        fpath = os.path.join(dirs.conn_train_dir, f"{summary_name}.csv")
        df = pd.read_csv(fpath)
        # df['train_exp'] = exp
        df_concat = pd.concat([df_concat, df])

    # rename cols
    cols = []
    for col in df_concat.columns:
        if "train" not in col:
            cols.append("train_" + col)
        else:
            cols.append(col)

    df_concat.columns = cols

    return df_concat

def eval_summary(summary_name="eval_summary"):
    """load eval summary containing all metrics about eval models.
    Summary across exps is concatenated and prefix 'eval' is appended to cols.
    Args:
        summary_name (str): name of summary file
    Returns:
        pandas dataframe containing concatenated exp summary
    """
    # look at model summary for eval results
    df_concat = pd.DataFrame()
    for exp in ["sc1", "sc2"]:
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

    return df_concat

def plot_train_predictions(dataframe, x='train_name', hue=None, x_order=None, hue_order=None):
    """plots training predictions (R CV) for all models in dataframe.
    Args:
        dataframe (pandas dataframe): must contain 'train_name' and 'train_R_cv'
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    # R
    sns.factorplot(x=x, y="train_R_cv", hue=hue, data=dataframe, order=x_order, hue_order=hue_order, legend=False, ci=None, size=4, aspect=2)
    plt.title("Model Training (CV Predictions)", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xticks(rotation="45", ha="right")
    plt.xlabel("")
    plt.ylabel("R", fontsize=20)
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_eval_predictions_all(dataframe, x='eval_name', hue=None, x_order=None, hue_order=None):
    """plots training predictions (R CV) for all models in dataframe.
    Args:
        dataframe (pandas dataframe): must contain 'train_name' and 'train_R_cv'
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    # R
    sns.factorplot(x=x, y="R_eval", hue=hue, data=dataframe, order=x_order, hue_order=hue_order, legend=False, ci=None, size=4, aspect=2)
    plt.title("Model Evaluation", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xticks(rotation="45", ha="right")
    plt.xlabel("")
    plt.ylabel("R", fontsize=20)
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_eval_predictions(dataframe, exp="sc1"):
    """plots evaluation predictions (R eval) for best model in dataframe for 'sc1' or 'sc2'
    Also plots model-dependent and model-independent noise ceilings.
    Args:
        dataframe (pandas dataframe): must contain 'train_name' and 'train_R_cv'
        exp (str): either 'sc1' or 'sc2'
        hue (str or None): default is 'eval_name'
    """
    # get best model (from train CV)
    best_model,_ = get_best_model(train_exp=exp)

    if exp is "sc1":
        eval_exp = "sc2"
    else:
        eval_exp = "sc1"

    dataframe = dataframe.query(f'eval_exp=="{eval_exp}" and eval_name=="{best_model}"')

    # get noise ceilings
    dataframe["eval_noiseceiling_Y"] = np.sqrt(dataframe.eval_noise_Y_R)
    dataframe["eval_noiseceiling_XY"] = np.sqrt(dataframe.eval_noise_Y_R) * np.sqrt(dataframe.eval_noise_X_R)

    # melt data into one column for easy plotting
    cols = ["eval_noiseceiling_Y", "eval_noiseceiling_XY", "R_eval"]
    df = pd.melt(dataframe, value_vars=cols, id_vars=set(dataframe.columns) - set(cols)).rename(
        {"variable": "data_type", "value": "data"}, axis=1
    )

    plt.figure(figsize=(8, 8))
    splot = sns.barplot(x="data_type", y="data", data=df)
    plt.title(f"Model Evaluation (exp={exp}: best model={best_model})", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel("")
    plt.ylabel("R", fontsize=20)
    plt.xticks(
        [0, 1, 2], ["noise ceiling (data)", "noise ceiling (model)", "model predictions"], rotation="45", ha="right"
    )
    # plt.legend(fontsize=15)

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

def map_eval(data="R", exp="sc1", model_name='best_model', colorbar=False, cscale=None, rois=True, atlas='MDTB_10Regions'):
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

    # get evaluation
    df_eval = eval_summary()

    # get best model
    model = model_name
    if model_name=="best_model":
        model,_ = get_best_model(train_exp=exp)

    # plot map
    fpath = os.path.join(dirs.conn_eval_dir, model)
    view = nio.view_cerebellum(gifti=os.path.join(fpath, f"group_{data}_vox.func.gii"), cscale=cscale, colorbar=colorbar)

    if rois:
        roi_summary(fpath=os.path.join(fpath, f"group_{data}_vox.nii"), atlas=atlas, plot=True)
    return view

def roi_summary(fpath, atlas='MDTB_10Regions', plot=True):
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
    rgba, cpal = nio.get_label_colors(fpath=atlas_dir + f'/{atlas}.label.gii')

    df_all = pd.DataFrame()
    data = cdata.read_suit_nii(fpath)
    roi_mean, regs = cdata.average_by_roi(data, rois)
    fname = Path(fpath).stem
    df1 = pd.DataFrame({'roi_mean': list(np.hstack(roi_mean)),
                    'regions': list(regs),
                    'fnames': np.repeat(fname, len(regs))})
    
    if plot:
        plt.figure()
        sns.barplot(x='regions', y='roi_mean', data=df1.query('regions!=0'), palette=cpal[1:])
        plt.xticks(rotation=45)
        plt.xlabel(atlas)
        plt.ylabel('ROI mean')

    return df1

def map_model_comparison(model_name, exp, method='subtract', colorbar=True, rois=True, atlas='MDTB_10Regions'):
    """plot surface map for best model
    Args:
    """

    # initialize directories
    dirs = const.Dirs(exp_name=exp)

    fpath = os.path.join(dirs.conn_eval_dir, 'model_comparison')
    fpath_gii = glob.glob(f'{fpath}/*{method}*{model_name}*.gii*')
    fpath_nii = glob.glob(f'{fpath}/*{method}*{model_name}*.nii*')

    view = nio.view_cerebellum(fpath_gii[0], cscale=None, colorbar=colorbar)

    if rois:
        roi_summary(fpath=fpath_nii[0], atlas=atlas, plot=True)
    
    return view

def map_weights(structure='cerebellum', exp='sc1', model_name='best_model', hemisphere='R', colorbar=False, cscale=None, rois=True, atlas='MDTB_10Regions'):
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
    
    if (rois) & (structure=='cerebellum'):
        roi_summary(fpath=os.path.join(fpath, f"group_weights_cerebellum.nii"), atlas=atlas, plot=True)
    return view

def map_atlas(fpath, structure='cortex'):
    """General purpose function for plotting *.label.gii or *.func.gii parcellations (cortex or cerebellum)
    Args: 
        fpath (str): full path to atlas
        anatomical_structure (str): default is 'cerebellum'. other options: 'cortex'
    Returns:
        viewing object to visualize parcellations
    """
    # initialize directory
    dirs = const.Dirs()

    if structure=='cerebellum':
        return nio.view_cerebellum(gifti=fpath) 
    elif structure=='cortex':
        return nio.view_cortex(gifti=fpath)
    else:
        print('please provide a valid parcellation')

def get_best_model(train_exp):
    """Get idx for best model based on either R_cv (or R_train)
    Args:
        exp (str): 'sc1' or 'sc2
    Returns:
        model name (str)
    """
    # load train summary (contains R CV of all trained models)
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.conn_train_dir, "train_summary.csv")
    df = pd.read_csv(fpath)

    # get mean values for each model
    tmp = df.groupby(["name", "X_data"]).mean().reset_index()

    # get best model (based on R CV or R train)
    try: 
        best_model = tmp[tmp["R_cv"] == tmp["R_cv"].max()]["name"].values[0]
        cortex = tmp[tmp["R_cv"] == tmp["R_cv"].max()]["X_data"].values[0]
    except:
        best_model = tmp[tmp["R_train"] == tmp["R_train"].max()]["name"].values[0]
        cortex = tmp[tmp["R_train"] == tmp["R_train"].max()]["X_data"].values[0]

    print(f"best model for {train_exp} is {best_model}")

    return best_model, cortex

def get_best_models(train_exp):
    """Get model_names, cortex_names for best models (NNLS, ridge, WTA) based on R_cv
    Args:
        exp (str): 'sc1' or 'sc2
    Returns:
        model_names (list of str), cortex_names (list of str)
    """
    # load train summary (contains R CV of all trained models)
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.conn_train_dir, "train_summary.csv")
    df = pd.read_csv(fpath)

    tmp = df.groupby(['X_data', 'model', 'hyperparameter', 'name']
                ).mean().reset_index()

    tmp1 = tmp.groupby(['X_data', 'model']
            ).apply(lambda x: x['R_cv'].max()
            ).reset_index(name='R_cv')

    tmp2 = tmp1.merge(tmp, on=['X_data', 'model', 'R_cv'])

    model_names = list(tmp2['name'])

    cortex_names = list(tmp2['X_data'])

    return model_names, cortex_names

def get_eval_models(exp):
    dirs = const.Dirs(exp_name=exp)
    df = pd.read_csv(os.path.join(dirs.conn_eval_dir, 'eval_summary.csv'))
    df = df[['name', 'X_data']].drop_duplicates() # get unique model names

    return df['name'].to_list(), np.unique(df['X_data'].to_list())

def train_weights(exp="sc1", model_name="ridge_tesselsWB162_alpha_6"):
    """gets training weights for a given model and summarizes into a dataframe
    averages the weights across the cerebellar voxels (1 x cortical ROIs)
    Args:
        exp (str): 'sc1' or 'sc2'
        model_name (str): default is 'ridge_tesselsWB162_alpha_6'
    Returns:
        pandas dataframe containing 'ROI', 'weights', 'subj', 'exp'
    """
    data_dict = defaultdict(list)
    dirs = const.Dirs(exp_name=exp)
    trained_models = glob.glob(os.path.join(dirs.conn_train_dir, model_name, "*.h5"))

    for fname in trained_models:

        # Get the model from file
        fitted_model = dd.io.load(fname)
        regex = r"_(s\d+)."
        subj = re.findall(regex, fname)[0]
        weights = np.nanmean(fitted_model.coef_, 0)

        data = {
            "ROI": np.arange(1, len(weights) + 1),
            "weights": weights,
            "subj": [subj] * len(weights),
            "exp": [exp] * len(weights),
        }

        # append data for each subj
        for k, v in data.items():
            data_dict[k].extend(v)

    return pd.DataFrame.from_dict(data_dict)

def plot_train_weights(dataframe, hue=None):
    """plots training weights in dataframe
    Args:
        dataframe (pandas dataframe): must contain 'ROI' and 'weights' cols
        hue (str or None): default is None
    """
    plt.figure(figsize=(8, 8))
    sns.lineplot(x="ROI", y="weights", hue=hue, data=dataframe, ci=None)

    exp = dataframe["exp"].unique()[0]

    plt.axhline(linewidth=2, color="r")
    plt.title(f"Cortical weights averaged across subjects for {exp}")
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel("# of ROIs", fontsize=20)
    plt.ylabel("Weights", fontsize=20)

def show_distance_matrix(roi='tessels0042'):
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