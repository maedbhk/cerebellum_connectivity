import os
import sys
import glob
import numpy as np
import json
import deepdish as dd
import pandas as pd
import connectivity.model as model
import connectivity.evaluation as ev
import timeit
import connectivity.io as cio
from connectivity.data import Dataset
import connectivity.constants as const

import warnings

warnings.filterwarnings("ignore")

np.seterr(divide="ignore", invalid="ignore")

"""
Created on Aug 31 11:14:19 2020
Main script for training and evaluating connectivity models

@authors: Maedbh King, Jörn Diedrichsen 
"""


def delete_conn_files():
    """delete any pre-existing connectivity output"""
    for exp in ["sc1", "sc2"]:
        dirs = const.Dirs(study_name=exp, glm=7)
        filelists = [glob.glob(os.path.join(dirs.con_train_dir, "*")), glob.glob(os.path.join(dirs.conn_eval_dir, "*"))]
        for filelist in filelists:
            for f in filelist:
                os.remove(f)
    print("deleting training and results data")


def get_default_train_config():
    # defaults training config:
    config = {
        "name": "ridge_tessels0162_A0",  # Model name - determines the directory
        "model": "L2regression",  # Model class name (must be in model.py)
        "param": {"alpha": 1},  # Parameter to model constructor
        "sessions": [1, 2],  # Sessions used for training data
        "glm": "glm7",  # GLM used for training data
        "train_exp": "sc1",  # Experiment used for training data
        "averaging": "sess",  # Avaraging scheme for X and Y (see data.py)
        "weighting": 2,  # 0: none, 1: by regr., 2: by full matrix
        "incl_inst": True,
        "X_data": "tessels0162",
        "Y_data": "cerebellum_suit",
        "subjects": ["s02", "s03","s04", "s06", "s08", "s09", "s10", "s12", "s14", "s15", "s17", "s18", "s19", "s20", "s21", "s22", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31"],
        "mode": "crossed",  # Training mode
    }
    return config


def get_default_eval_config():
    # defaults training config:
    config = {
        "name": "ridge_tessels0162_A0",
        "sessions": [1, 2],
        "glm": "glm7",
        "train_exp": "sc1",
        "eval_exp": "sc2",
        "averaging": "sess",
        "weighting": 2,  # 0: none, 1: by regr., 2: by full matrix
        "incl_inst": True,
        "X_data": "tessels0162",
        "Y_data": "cerebellum_suit",
        "subjects": ["s02", "s03","s04", "s06", "s08", "s09", "s10", "s12", "s14", "s15", "s17", "s18", "s19", "s20", "s21", "s22", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31"],
        "mode": "crossed",
        "eval_splitby": None,
    }
    return config


def train_models(config, save=False):
    """
    train_model: trains a specific model class on X and Y data from a specific experiment for all subjects
    Parameters:
        config (dict)
            Training configuration (see get_default_train_config)
        save (bool)
            Save fitted models automatically to conn-folder
    Returns:
        models (list)
            List of trained models for all subject
    """
    exp = config["train_exp"]
    dirs = const.Dirs(exp_name=exp, glm=config["glm"])
    models = []

    # Store the training configuration in model directory
    if save:
        fname = [config["name"]]
        fpath = dirs.conn_train_dir / config["name"]
        if not os.path.exists(fpath):
            print(f"creating {fpath}")
            os.makedirs(fpath)
        cname = fpath / "train_config.json"
        with open(cname, "w") as fp:
            json.dump(config, fp)

    # Loop over subjects and train
    for s in config["subjects"]:
        print(f"Subject " + s + "\n")

        # Get the condensed data
        Ydata = Dataset(glm=config["glm"], subj_id=s, roi=config["Y_data"])
        Ydata.load()
        Y, T = Ydata.get_data(averaging=config["averaging"], weighting=config["weighting"])
        Xdata = Dataset(glm=config["glm"], subj_id=s, roi=config["X_data"])
        Xdata.load()
        X, T = Xdata.get_data(averaging=config["averaging"], weighting=config["weighting"])
        # Generate new model and put in the list
        newModel = getattr(model, config["model"])(**config["param"])
        models.append(newModel)
        # Fit this model
        tic = timeit.default_timer() 
        models[-1].fit(X, Y)
        toc = timeit.default_timer()
        models[-1].fit_time=toc-tic

        # Save the fitted model to disk if required
        if save:
            fname = _get_model_name(config["name"], config["train_exp"], s)
            dd.io.save(fname, models[-1], compression=None)

    # Return list of models
    return models


def eval_models(config):
    """
    eval_models: trains a specific model class on X and Y data from a specific experiment for all subjects
    Parameters:
        config (dict)
            Eval configuration (see get_default_eval_config
    Returns:
        T (pd.DataFrame)
            Evaluation of different models on the data
    """

    D = pd.DataFrame()

    for i, s in enumerate(config["subjects"]):
        print(f"Evaluating Subject {s}\n")

        # Get the data
        Ydata = Dataset(experiment=config["eval_exp"], glm=config["glm"], subj_id=s, roi=config["Y_data"])
        Ydata.load()
        Y, T = Ydata.get_data(averaging=config["averaging"], weighting=config["weighting"])
        Xdata = Dataset(experiment=config["eval_exp"], glm=config["glm"], subj_id=s, roi=config["X_data"])
        Xdata.load()
        X, T = Xdata.get_data(averaging=config["averaging"], weighting=config["weighting"])

        # Get the model from file
        fname = _get_model_name(config["name"], config["train_exp"], s)
        fitted_model = dd.io.load(fname)

        # Save the fitted model to disk if required
        Ypred = fitted_model.predict(X)
        if config["mode"] == "crossed":
            Ypred = np.r_[Ypred[T.sess == 2, :], Ypred[T.sess == 1, :]]

        # Add the subject number
        D.loc[i, "SN"] = s

        # Copy over all scalars or strings to the Data frame:
        for key, value in config.items():
            if type(value) is not list:
                D.loc[i, key] = value

        # Add the evaluation
        D.loc[i, "R"], Rvox = ev.calculate_R(Y, Ypred)  # R between predicted and observed
        D.loc[i, "R2"], R2vox = ev.calculate_R2(Y, Ypred)  # R2 between predicted and observed
        D.loc[i, "noise_Y_R"], _, D.loc[i, "noise_Y_R2"], _ = ev.calculate_reliability(Y, T)  # Noise ceiling for cerebellum (squared)
        D.loc[i, "noise_X_R"], _, D.loc[i, "noise_X_R2"], _ = ev.calculate_reliability(Ypred, T)  # Noise ceiling for cortex (squared)
        pass

    # Return list of models
    return D

def average_models(name,exp,subj_id=const.return_subjs):
    """
        Returns a model with connectivity weights averaged across subjects 

    Args:
        name ([str]): Name of the trained model
        exp ([str]): "sc1" or "sc2"
        subj_id ([list], optional): Lists of subjects, Defaults to const.return_subjs.
    """
    num_subj = len(subj_id)
    for i,s in enumerate(subj_id):
        fname = _get_model_name(name, exp, s)
        M = dd.io.load(fname)
        if i==0:
            W = np.zeros((num_subj,M.coef_.shape[0],M.coef_.shape[1]))
        W[i,:,:] = M.coef_
    M.coef_ = np.nanmean(W,axis=0)
    return M

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
