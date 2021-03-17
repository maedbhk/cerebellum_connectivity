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
import nibabel as nib

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


def plot_train_predictions(dataframe, hue=None):
    """plots training predictions (R CV) for all models in dataframe.

    Args:
        dataframe (pandas dataframe): must contain 'train_name' and 'train_R_cv'
        hue (str or None): can be 'train_exp', 'Y_data' etc.
    """
    plt.figure(figsize=(15, 10))
    # R
    sns.factorplot(x="train_name", y="train_R_cv", hue=hue, data=dataframe, legend=False, ci=None, size=4, aspect=2)
    plt.title("Model Training (CV Predictions)", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xticks(rotation="45", ha="right")
    plt.xlabel("")
    plt.ylabel("R", fontsize=20)
    plt.legend(fontsize=15)


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


def plot_eval_map(gifti_func="group_R_vox", exp="sc1", model=None, cscale=None):
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
    surf_data = nio.nib_load(os.path.join(dirs.conn_eval_dir, model, f"{gifti_func}.func.gii"))
    view = nilearn_flatmap(surf_data.darrays[0].data, cscale=cscale) #symmetric_cmap=False,
    return view


def plot_train_map(gifti_func='group_weights_cerebellum', exp='sc1', model=None, cscale=None):
    # initialise directories
    dirs = const.Dirs(exp_name=exp)

    # get evaluation
    df_eval = eval_summary()

    # get best model
    if not model:
        model = get_best_model(train_exp=exp)
    
    # plot map
    surf_data = nio.nib_load(os.path.join(dirs.conn_train_dir, model, f"{gifti_func}.func.gii"))
    view = nilearn_flatmap(surf_data.darrays[0].data, cscale=cscale) #symmetric_cmap=False,
    return view


def nilearn_flatmap(data, cmap='jet', threshold=None, bg_map=None, cscale=None):

    # full path to surface
    surf_dir = os.path.join(flatmap._surf_dir,'FLAT.surf.gii')

    # load topology
    flatsurf = nib.load(surf_dir)
    vertices = flatsurf.darrays[0].data
    faces    = flatsurf.darrays[1].data

    # Determine underlay and assign color
    # underlay = nib.load(underlay)

    # Determine scale
    if cscale is None:
        cscale = [data.min(), data.max()]

    # nilearn seems to
    view = view_surf([vertices,faces], data, bg_map=bg_map, cmap=cmap,
                        threshold=threshold, vmin=cscale[0], vmax=cscale[1])
    return view


def get_best_model(train_exp):
    """Get idx for best ridge based on either rmse_train or rmse_cv.

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
    best_model = tmp[tmp["R_cv"] == tmp["R_cv"].max()]["name"].values[0]

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
