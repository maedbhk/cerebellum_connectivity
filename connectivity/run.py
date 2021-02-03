import os
import sys
import glob
import numpy as np
import json
import deepdish as dd
import pandas as pd
import connectivity.model as model
import connectivity.evaluation as ev
import copy

from sklearn.model_selection import cross_val_score

from collections import defaultdict

import connectivity.io as cio
from connectivity.data import Dataset
import connectivity.constants as const

import warnings

warnings.filterwarnings("ignore")

np.seterr(divide="ignore", invalid="ignore")

"""Main module for training and evaluating connectivity models.

   @authors: Maedbh King, Jörn Diedrichsen  

  Typical usage example:

  config = get_default_train_config()
  models = train_models(config=config)

  config = get_default_eval_config()
  models = eval_models(config)
"""


def delete_conn_files():
    """delete any pre-existing connectivity output."""
    for exp in ["sc1", "sc2"]:
        dirs = const.Dirs(study_name=exp, glm=7)
        filelists = [
            glob.glob(os.path.join(dirs.conn_train_dir, "*")),
            glob.glob(os.path.join(dirs.conn_eval_dir, "*")),
        ]
        for filelist in filelists:
            for f in filelist:
                os.remove(f)
    print("deleting training and results data")


def get_default_train_config():
    """Defaults for training model(s).

    Returns:
        A dict mapping keys to training parameters.
    """
    # defaults training config:
    config = {
        "name": "L2_WB162_A1",  # Model name - determines the directory
        "model": "L2regression",  # Model class name (must be in model.py)
        "param": {"alpha": 1},  # Parameter to model constructor
        "sessions": [1, 2],  # Sessions used for training data
        "glm": "glm7",  # GLM used for training data
        "train_exp": "sc1",  # Experiment used for training data
        "averaging": "sess",  # Averaging scheme for X and Y (see data.py)
        "weighting": True,  # "none" "full" "diag"
        # "incl_inst": True,
        "X_data": "tesselsWB162",
        "Y_data": "cerebellum_grey",
        "validate_model": False,
        "cv_fold": None,
        "subjects": [
            "s01",
            "s03",
            "s04",
            "s06",
            "s08",
            "s09",
            "s10",
            "s12",
            "s14",
            "s15",
            "s17",
            "s18",
            "s19",
            "s20",
            "s21",
            "s22",
            "s24",
            "s25",
            "s26",
            "s27",
            "s28",
            "s29",
            "s30",
            "s31",
        ],
        "mode": "crossed",  # Training mode
    }
    return config


def get_default_eval_config():
    """Defaults for evaluating model(s).

    Returns:
        A dict mapping keys to evaluation parameters.
    """
    config = {
        "name": "L2_WB162_A1",
        "sessions": [1, 2],
        "glm": "glm7",
        "train_exp": "sc1",
        "eval_exp": "sc2",
        "averaging": "sess",
        "weighting": True,  # 0: none, 1: by regr., 2: by full matrix???
        "incl_inst": True,
        "X_data": "tesselsWB162",
        "Y_data": "cerebellum_grey",
        "subjects": [
            "s01",
            "s03",
            "s04",
            "s06",
            "s08",
            "s09",
            "s10",
            "s12",
            "s14",
            "s15",
            "s17",
            "s18",
            "s19",
            "s20",
            "s21",
            "s22",
            "s24",
            "s25",
            "s26",
            "s27",
            "s28",
            "s29",
            "s30",
            "s31",
        ],
        "mode": "crossed",
        "eval_splitby": None,
        "save_voxels": False,
    }
    return config


def train_models(config, save=False):
    """Trains a specific model class on X and Y data from a specific experiment for subjects listed in config.

    Args:
        config (dict): Training configuration, returned from get_default_train_config()
        save (bool): Optional; Save fitted models automatically to disk.
    Returns:
        models (list): list of trained models for subjects listed in config.
    """

    dirs = const.Dirs(exp_name=config["train_exp"], glm=config["glm"])
    models = []
    # Store the training configuration in model directory
    if save:
        fpath = dirs.conn_train_dir / config["name"]
        if not os.path.exists(fpath):
            print(f"creating {fpath}")
            os.makedirs(fpath)

        cname = fpath / "train_config.json"
        with open(cname, "w") as fp:
            json.dump(config, fp)

    # Loop over subjects and train
    for subj in config["subjects"]:
        print(f"Training model on {subj}")

        # get data
        Y, Y_info, X, X_info = _get_data(config=config, subj=subj)

        # Generate new model and put in the list
        new_model = getattr(model, config["model"])(**config["param"])
        models.append(new_model)

        # Fit model and get train rmse
        models[-1].fit(X, Y)
        Y_pred = models[-1].predict(X, Y)
        models[-1]['train_rmse'] = ev.calculate_rmse(Y, Y_pred)

        # get cv rmse
        if config['validate_model']: 
            cv_rmse_all = cross_val_score(models[-1], X, Y, scoring=ev.calculate_rmse, cv=config['cv_fold'])
            models[-1]['cv_rmse'] = np.nanmean(cv_rmse_all)

        # Save the fitted model to disk if required
        if save:
            fname = _get_model_name(
                train_name=config["name"], exp=config["train_exp"], subj_id=subj
            )
            dd.io.save(fname, models[-1], compression=None)

    return models


def eval_models(config):
    """Evaluates a specific model class on X and Y data from a specific experiment for subjects listed in config.

    Args:
        config (dict): Evaluation configuration, returned from get_default_eval_config()
    Returns:
        models (pd dataframe): evaluation of different models on the data
    """

    eval_all = defaultdict(list)
    eval_voxels = defaultdict(list)

    for idx, subj in enumerate(config["subjects"]):

        print(f"Evaluating model on {subj}")

        # get data
        Y, Y_info, X, X_info = _get_data(config=config, subj=subj)

        # Get the model from file
        fname = _get_model_name(
            train_name=config["name"], exp=config["train_exp"], subj_id=subj
        )
        fitted_model = dd.io.load(fname)

        # Get model predictions
        Y_pred = fitted_model.predict(X)
        if config["mode"] == "crossed":
            Y_pred = np.r_[y_pred[Y_info.sess == 2, :], Y_pred[Y_info.sess == 1, :]]

        # Add the subject number
        eval_all["subj_id"].append(subj)

        # Copy over all scalars or strings to eval_all dataframe:
        for key, value in config.items():
            if type(value) is not list:
                # df.loc[idx, key] = value
                eval_all[key].append(value)

        # add evaluation (summary)
        eval_data = _get_eval(Y=Y, Y_pred=Y_pred, Y_info=Y_info, X_info=X_info)
        for k, v in eval_data.items():
            eval_all[k].append(v)

        # add evaluation (voxels)
        if config["save_voxels"]:
            for k, v in eval_data.items():
                if 'vox' in k:
                    eval_voxels[k].append(v)
        else:
            eval_all = [eval_all[k] for k in eval_all if 'vox' not in k]

    # Return list of models
    return pd.DataFrame.from_dict(eval_all), eval_voxels


def _get_eval(Y, Y_pred, Y_info, X_info):
    """Compute evaluation, returning summary and voxel data.

    Args:
        Y (np array):
        Y_pred (np array):
        Y_info (pd dataframe):
        X_info (pd dataframe):
    Returns:
        dict containing evaluations (R, R2, noise).
    """
    # initialise dictionary
    data = {}

    # Add the evaluation
    data["R"], data["R_vox"]  = ev.calculate_R(Y=Y, Y_pred=Y_pred)

    # R between predicted and observed
    data["R2"], data["R2_vox"] = ev.calculate_R2(Y=Y, Y_pred=Y_pred)

    # R2 between predicted and observed
    data["noise_Y_R"], data["noise_Y_R_vox"], data["noise_Y_R2"], data["noise_Y_R2_vox"] = ev.calculate_reliability(
        Y=Y, dataframe=Y_info
    )

    # Noise ceiling for cerebellum (squared)
    data["noise_X_R"], data["noise_X_R_vox"], data["noise_X_R2"], data["noise_X_R2_vox"] = ev.calculate_reliability(
        Y=Y_pred, dataframe=X_info
    )

    # # Noise ceiling for cortex (squared)
    #     pass

    return data


def _get_data(config, subj):
    # Get the data
    Ydata = Dataset(
        experiment=config["eval_exp"],
        glm=config["glm"],
        subj_id=subj,
        roi=config["Y_data"],
    )

    # load mat
    Ydata.load_mat()

    Y, Y_info = Ydata.get_data(
        averaging=config["averaging"], weighting=config["weighting"]
    )

    Xdata = Dataset(
        experiment=config["eval_exp"],
        glm=config["glm"],
        subj_id=subj,
        roi=config["X_data"],
    )

    # load mat
    Xdata.load_mat()

    X, X_info = Xdata.get_data(
        averaging=config["averaging"], weighting=config["weighting"]
    )

    return Y, Y_info, X, X_info


def _get_model_name(train_name, exp, subj_id):
    """returns path/name for connectivity training model outputs.

    Args:
        train_name (str): Name of trained model
        exp (str): Experiment name
        subj_id (str): Subject id
    Returns:
        fpath (str): Full path and name to connectivity output for model training.
    """

    dirs = const.Dirs(exp_name=exp)
    fname = f"{train_name}_{subj_id}.h5"
    fpath = os.path.join(dirs.conn_train_dir, train_name, fname)
    return fpath
