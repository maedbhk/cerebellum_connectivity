import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
import glob
import deepdish as dd
from collections import defaultdict
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
from nilearn.plotting import view_surf
from nilearn.surface import load_surf_data
import nibabel as nib

import connectivity.data as cdata
import connectivity.sparsity as csparsity
import connectivity.constants as const
import connectivity.nib_utils as nio

plt.rcParams["axes.grid"] = False


def train_summary(summary_name="train_summary", save_online=True):
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

    if save_online:
        log_online(dataframe=df_concat)

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


def log_online(dataframe):
    # group df by train name and take first row (not interested in results here)
    dataframe = dataframe.groupby('train_name').first().reset_index()[['train_name', 'train_exp', 'train_X_data', 'train_Y_data', 'train_model', 'train_glm','train_averaging']]

    dirs = const.Dirs()
    dataframe.to_csv(os.path.join(dirs.base_dir, 'model_logs.csv'))


def plot_train_predictions(dataframe, x='train_name', hue=None, x_order=None, hue_order=None):
    """plots training predictions (R CV) for all models in dataframe.

    Args:
        dataframe (pandas dataframe): must contain 'train_name' and 'train_R_cv'
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    plt.figure(figsize=(15, 10))
    # R
    sns.factorplot(x=x, y="train_R_cv", hue=hue, data=dataframe, order=x_order, hue_order=hue_order, legend=False, size=4, aspect=2)
    plt.title("Model Training (CV Predictions)", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xticks(rotation="45", ha="right")
    plt.xlabel("")
    plt.ylabel("R", fontsize=20)
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_eval_predictions(dataframe, exp="sc1"):
    """plots evaluation predictions (R eval) for best model in dataframe for 'sc1' or 'sc2'

    Also plots model-dependent and model-independent noise ceilings.

    Args:
        dataframe (pandas dataframe): must contain 'train_name' and 'train_R_cv'
        exp (str): either 'sc1' or 'sc2'
        hue (str or None): default is 'eval_name'
    """
    # get best model (from train CV)
    best_model = get_best_model(train_exp=exp)

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


def plot_eval_map(gifti_func="group_R_vox", exp="sc1", model=None, cscale=None, symmetric_cmap=False):
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
    if not model:
        model = get_best_model(train_exp=exp)

    # plot map
    surf_data = os.path.join(dirs.conn_eval_dir, model, f"{gifti_func}.func.gii")
    view = nio.view_cerebellum(data=surf_data, cscale=cscale, symmetric_cmap=symmetric_cmap) #symmetric_cmap=False,
    return view


def plot_ntakeall_map(roi='tessels0042', n=2, model_ext='positive', hem='L', metric=None, exp='sc1', cscale=None, vmin=None, symmetric_cmap=False):
    """Plot NTakeAll map for `roi` and `exp`
    
    Args: 
        roi (str): 'tessels0042', 'tessels1002' etc.
        n (int): 2,3,4 etc.
        exp (str): 'sc1' or 'sc2'
        model_ext (str): 'positive' or 'absolute'
        hem (str): 'L', 'R', 'L_R'
        cscale (bool): default is None
        symmetric_cmap (bool): default is False
    Returns: 
        Returns view object for visualizing winner map
    """
    dirs = const.Dirs(exp_name=exp)

    if metric:
        surf_fname = os.path.join(dirs.conn_train_dir, f'NTakeAll_{roi}_{n}_{model_ext}', f'group_ntakeall_cerebellum_{hem}_{metric}.func.gii')
    else:
        surf_fname = os.path.join(dirs.conn_train_dir, f'NTakeAll_{roi}_{n}_{model_ext}', f'group_ntakeall_cerebellum_{hem}.func.gii')
    view = nio.view_cerebellum(data=surf_fname, cscale=cscale, symmetric_cmap=symmetric_cmap, vmin=vmin)

    return view


def plot_train_map(gifti_func='group_weights_cerebellum', exp='sc1', model=None, cscale=None, hemisphere='R', symmetric_cmap=False):
    # initialise directories
    dirs = const.Dirs(exp_name=exp)

    # get evaluation
    df_eval = eval_summary()

    # get best model
    if not model:
        model = get_best_model(train_exp=exp)
    
    # plot either cerebellum or cortex
    if 'cerebellum' in gifti_func:
        surf_fname = os.path.join(dirs.conn_train_dir, model, f"{gifti_func}.func.gii")
        view = nio.view_cerebellum(data=surf_fname, cscale=cscale, symmetric_cmap=symmetric_cmap)
    elif 'cortex' in gifti_func:
        if hemisphere=='R':
            hem_name = 'CortexRight'
        elif hemisphere=='L':
            hem_name = 'CortexLeft'
        surf_fname = os.path.join(dirs.conn_train_dir, model, f"{gifti_func}.{hem_name}.func.gii")
        view = nio.view_cortex(data=surf_fname, cscale=cscale, hemisphere=hemisphere, symmetric_cmap=symmetric_cmap)
    else:
        print("gifti must contain either cerebellum or cortex in name")
    return view


def plot_winner_map(roi='tessels0042', exp='sc1', cscale=None, symmetric_cmap=False):
    """Plot winner-take-all map for `roi` for `exp`

    Args: 
        roi (str): 'tessels0042', 'tessels1002' etc.
        exp (str): 'sc1' or 'sc2'
        cscale (bool): default is None
        symmetric_cmap (bool): default is False
    Returns: 
        Returns view object for visualizing winner map
    """
    dirs = const.Dirs(exp_name=exp)

    surf_fname = os.path.join(dirs.conn_train_dir, f'WTA_{roi}', "group_wta_cerebellum.label.gii")
    view = nio.view_cerebellum(data=surf_fname, cscale=cscale, symmetric_cmap=symmetric_cmap)
    surf_data = load_surf_data(surf_fname)
    # view = flatmap.plot(surf_data, overlay_type='label')

    return view


def get_best_model(train_exp):
    """Get idx for best ridge based on R_cv
    If crossvalidation wasn't done, then R_train is used

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
    tmp = df.groupby("name").mean().reset_index()

    # get best model (based on R CV)
    try:
        best_model = tmp[tmp["R_cv"] == tmp["R_cv"].max()]["name"].values[0]
    except:
        best_model = tmp[tmp["R_train"] == tmp["R_train"].max()]["name"].values[0]

    print(f"best model for {train_exp} is {best_model}")

    return best_model


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


def plot_parcellation(parcellation=None, anatomical_structure='cerebellum', hemisphere=None, subset=None, colorbar=None):
    """General purpose function for plotting parcellations (cortex or cerebellum)

    Args: 
        parcellation (str):  any of the following: 'yeo7', 'yeo17', 'MDTB_10', 'tessels<num>', 'buckner7', 'buckner17'
        anatomical_structure (str): default is 'cerebellum'. other options: 'cortex'
        hemisphere (None or str): default is None. other options are 'L' and 'R'
    Returns:
        viewing object to visualize parcellations
    """
    # initialize directory
    dirs = const.Dirs()

    if parcellation=='MDTB_10':
        surf_labels = os.path.join(flatmap._surf_dir,'MDTB_10Regions.label.gii')
    elif parcellation=='yeo7':
        surf_labels = os.path.join(dirs.fs_lr_dir, f'Yeo_JNeurophysiol11_7Networks.32k.{hemisphere}.label.gii')
    elif parcellation=='yeo17':
        surf_labels = os.path.join(dirs.fs_lr_dir, f'Yeo_JNeurophysiol11_17Networks.32k.{hemisphere}.label.gii')
    elif parcellation=='buckner7':
        surf_labels = os.path.join(flatmap._surf_dir,'Buckner_7Networks.label.gii')
    elif parcellation=='buckner17':
        surf_labels = os.path.join(flatmap._surf_dir,'Buckner_17Networks.label.gii')
    elif 'tessels' in parcellation:
        parcellation = ''.join(re.findall(r'[0-9]', parcellation))
        surf_labels = os.path.join(dirs.fs_lr_dir, f'Icosahedron-{parcellation}.32k.{hemisphere}.label.gii')
    else:
        print('please provide a valid parcellation')
    
    if anatomical_structure=='cerebellum':
        return nio.view_cerebellum(data=surf_labels)
    elif anatomical_structure=='cortex':
        return nio.view_cortex(data=surf_labels, hemisphere=hemisphere, subset=subset, colorbar=colorbar)
    else:
        print("please provide a valid anatomical structure, either 'cerebellum' or 'cortex'")


def plot_distance_matrix(roi='tessels0042', hemisphere='R'):
    """Plot matrix of distances for cortical `roi` and `hemisphere`

    Args: 
        roi (str): default is 'tessels0042'
        hemisphere (str): 'R' or 'L'
    Returns: 
        plots distance matrix
    """

    # get labels for `hemisphere`
    labels = csparsity.get_labels_hemisphere(roi, hemisphere)

    # get distances for `roi` and `hemisphere`
    distances = cdata.get_distance_matrix(roi=roi)[0]
    distances = distances[labels,][:, labels]

    # visualize matrix of distances
    plt.imshow(distances)
    plt.colorbar()
    plt.show()