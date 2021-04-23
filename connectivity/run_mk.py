import os
import sys
import glob
import numpy as np
import json
import deepdish as dd
import pandas as pd
import copy
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import connectivity.io as cio
from connectivity import data as cdata
import connectivity.constants as const
import connectivity.sparsity as csparsity
import connectivity.model as model
import connectivity.evaluation as ev
import connectivity.nib_utils as nio

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
        "cv_fold": None, #TO IMPLEMENT: "sess", "run" (None is "tasks")
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
        "save_weights": False, #Training mode
    }
    return config


def get_default_eval_config():
    """Defaults for evaluating model(s).

    Returns:
        A dict mapping keys to evaluation parameters.
    """
    config = {
        "name": "L2_tessels1002_A1",
        "sessions": [1, 2],
        "glm": "glm7",
        "train_exp": "sc1",
        "eval_exp": "sc2",
        "averaging": "sess",
        "weighting": True,  # 0: none, 1: by regr., 2: by full matrix???
        "incl_inst": True,
        "X_data": "tessels1002",
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
        "splitby": None,
        "save_maps": False,
        "threshold": 0.1
    }
    return config


def train_models(config, save=False):
    """Trains a specific model class on X and Y data from a specific experiment for subjects listed in config.

    Args:
        config (dict): Training configuration, returned from get_default_train_config()
        save (bool): Optional; Save fitted models automatically to disk.
    Returns:
        models (list): list of trained models for subjects listed in config.
        train_all (pd dataframe): dataframe containing
    """

    dirs = const.Dirs(exp_name=config["train_exp"], glm=config["glm"])
    models = []
    train_all = defaultdict(list)
    # Store the training configuration in model directory
    if save:
        fpath = os.path.join(dirs.conn_train_dir, config["name"])
        cio.make_dirs(fpath)
        cio.save_dict_as_JSON(os.path.join(fpath, "train_config.json"), config)

    # Loop over subjects and train
    for subj in config["subjects"]:
        print(f"Training model on {subj}")

        # get data
        Y, Y_info, X, X_info = _get_data(config=config, exp=config["train_exp"], subj=subj)

        # Generate new model and put in the list
        new_model = getattr(model, config["model"])(**config["param"])
        models.append(new_model)

        # cross the sessions
        if config["mode"] == "crossed":
            Y = np.r_[Y[Y_info.sess == 2, :], Y[Y_info.sess == 1, :]]

        # Fit model, get train and validate metrics
        models[-1].fit(X, Y)
        models[-1].rmse_train, models[-1].R_train = train_metrics(models[-1], X, Y)

        # collect train metrics (rmse and R)
        data = {
            "subj_id": subj,
            "rmse_train": models[-1].rmse_train,
            "R_train": models[-1].R_train
            }

        # run cross validation and collect metrics (rmse and R)
        if config['validate_model']:
            models[-1].rmse_cv, models[-1].R_cv = validate_metrics(models[-1], X, Y, X_info, config["cv_fold"])
            data.update({"rmse_cv": models[-1].rmse_cv,
                        "R_cv": models[-1].R_cv
                        })

        # Copy over all scalars or strings from config to eval dict:
        for key, value in config.items():
            if not isinstance(value, (list, dict)):
                data.update({key: value})

        for k, v in data.items():
            train_all[k].append(v)

        # Save the fitted model to disk if required
        if save:
            fname = _get_model_name(train_name=config["name"], exp=config["train_exp"], subj_id=subj)
            dd.io.save(fname, models[-1], compression=None)

    return models, pd.DataFrame.from_dict(train_all)


def train_metrics(model, X, Y):
    """computes training metrics (rmse and R) on X and Y

    Args:
        model (class instance): must be fitted model
        X (nd-array):
        Y (nd-array):
    Returns:
        rmse_train (scalar), R_train (scalar)
    """
    Y_pred = model.predict(X)

    # get train rmse and R
    rmse_train = mean_squared_error(Y, Y_pred, squared=False)
    R_train, _ = ev.calculate_R(Y, Y_pred)

    return rmse_train, R_train


def validate_metrics(model, X, Y, X_info, cv_fold):
    """computes CV training metrics (rmse and R) on X and Y

    Args:
        model (class instance): must be fitted model
        X (nd-array):
        Y (nd-array):
        cv_fold (int): number of CV folds
    Returns:
        rmse_cv (scalar), R_cv (scalar)
    """
    # get cv rmse and R
    rmse_cv_all = np.sqrt(cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv=cv_fold) * -1)
    # TO DO: implement train/validate splits for "sess", "run"
    r_cv_all = cross_val_score(model, X, Y, scoring=ev.calculate_R_cv, cv=cv_fold)

    return np.nanmean(rmse_cv_all), np.nanmean(r_cv_all)


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
        Y, Y_info, X, X_info = _get_data(config=config, exp=config["eval_exp"], subj=subj)

        # Get the model from file
        fname = _get_model_name(train_name=config["name"], exp=config["train_exp"], subj_id=subj)
        fitted_model = dd.io.load(fname)

        # Get model predictions
        Y_pred = fitted_model.predict(X)
        if config["mode"] == "crossed":
            Y_pred = np.r_[Y_pred[Y_info.sess == 2, :], Y_pred[Y_info.sess == 1, :]]

        # get rmse
        rmse = mean_squared_error(Y, Y_pred, squared=False)
        data = {"rmse_eval": rmse, "subj_id": subj}

        # Copy over all scalars or strings to eval_all dataframe:
        for key, value in config.items():
            if type(value) is not list:
                data.update({key: value})

        # add evaluation (summary)
        evals = _get_eval(Y=Y, Y_pred=Y_pred, Y_info=Y_info, X_info=X_info)
        data.update(evals)

        # add sparsity metric (voxels)
        sparsity_results = _get_sparsity(config, fitted_model)
        data.update(sparsity_results)

        # add evaluation (voxels)
        if config["save_maps"]:
            for k, v in data.items():
                if "vox" in k:
                    eval_voxels[k].append(v)

        # don't save voxel data to summary
        data = {k: v for k, v in data.items() if "vox" not in k}

        # append data for each subj
        for k, v in data.items():
            eval_all[k].append(v)
    
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
    data["R_eval"], data["R_vox"] = ev.calculate_R(Y=Y, Y_pred=Y_pred)

    # R between predicted and observed
    data["R2"], data["R2_vox"] = ev.calculate_R2(Y=Y, Y_pred=Y_pred)

    # R2 between predicted and observed
    (
        data["noise_Y_R"],
        data["noise_Y_R_vox"],
        data["noise_Y_R2"],
        data["noise_Y_R2_vox"],
    ) = ev.calculate_reliability(Y=Y, dataframe=Y_info)

    # Noise ceiling for cerebellum (squared)
    (
        data["noise_X_R"],
        data["noise_X_R_vox"],
        data["noise_X_R2"],
        data["noise_X_R2_vox"],
    ) = ev.calculate_reliability(Y=Y_pred, dataframe=X_info)

    # calculate noise ceiling
    data["noiseceiling_Y_R_vox"] = np.sqrt(data["noise_Y_R_vox"])
    data["noiseceiling_XY_R_vox"] = np.sqrt(data["noise_Y_R_vox"] * np.sqrt(data["noise_X_R_vox"]))

    # # Noise ceiling for cortex (squared)
    #     pass

    return data


def _get_sparsity_OLD(config, fitted_model):
    """Get sparsity metrics for fitted model

    Args: 
        config (dict): must contain 'X_data', 'threshold'
        fitted_model (obj): must contain 'coef_'
    Returns: 
        data_all (dict): fields are different sparsity metrics
    """

    data_all = defaultdict(list)
    # loop over hemispheres
    for hem in ['L', 'R']:
    
        # get labels for `roi` and `hemisphere`
        labels = csparsity.get_labels_hemisphere(roi=config['X_data'], hemisphere=hem)
        
        # get distances for `roi` and `hemisphere`
        distances = cdata.get_distance_matrix(roi=config['X_data'])[0]
        distances = distances[labels,][:, labels]

        # get weights for hemisphere
        weights = fitted_model.coef_[:, labels]
        
        # sort weights and return indices for top `threshold`
        weight_indices = csparsity.threshold_weights(weights=weights, threshold=config['threshold'])

        # calculate sum/var/std of distances for weights
        sparsity_dict = csparsity.get_distance_weights(weight_indices, distances)

        # calculate coefficients weighted by distance and add to dict
        sparsity_dict.update(csparsity.weight_distances(weights=weights, distances=distances))

        # append data for each hemisphere
        for k, v in sparsity_dict.items():
            data_all[k].append(v)
    
    # get average across hemispheres
    for k, v in data_all.items():
        data_all[k] = np.nanmean(np.stack(v, axis=1), axis=1)
    
    return data_all


def _get_data(config, exp, subj):
    """get X and Y data for exp and subj

    Args:
        config (dict): must contain keys for glm, Y_data, X_data, averaging, weighting
        exp (str): 'sc1' or 'sc2'
        subj (str): default subjs are set in constants.py
    Returns:
        Y (nd array), Y_info (pd dataframe), X (nd array), X_info (pd dataframe)
    """
    # Get the data
    Ydata = cdata.Dataset(
        experiment=exp,
        glm=config["glm"],
        subj_id=subj,
        roi=config["Y_data"],
    )

    # load mat
    Ydata.load_mat()

    Y, Y_info = Ydata.get_data(averaging=config["averaging"], weighting=config["weighting"])

    Xdata = cdata.Dataset(
        experiment=exp,
        glm=config["glm"],
        subj_id=subj,
        roi=config["X_data"],
    )

    # load mat
    Xdata.load_mat()

    X, X_info = Xdata.get_data(averaging=config["averaging"], weighting=config["weighting"])

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
