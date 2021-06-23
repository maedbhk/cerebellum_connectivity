# import libraries
import os, shutil
# import click
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import deepdish as dd

import copy

from scipy.stats import mode
from random import seed, sample
from collections import defaultdict
from pathlib import Path
import SUITPy.flatmap as flatmap

import connectivity.constants as const
import connectivity.io as cio
import connectivity.nib_utils as nio
from connectivity.data import Dataset
from connectivity import data as cdata
import connectivity.model as model
import connectivity.run as run
import connectivity.run_mk as run_connect
from connectivity import visualize as summary

import connectivity.evaluation as ev

from sklearn.cluster import KMeans

import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib tk


# splitting subjects for discovery and replication set
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

    print(f"splitting subjects:")   
    # set random seed
    seed(1)

    # get number of subjects in test (round down)
    num_in_test = int(np.floor(test_size * len(subj_ids)))

    # select test set
    test_subjs = list(sample(subj_ids, num_in_test))
    train_subjs = list([x for x in subj_ids if x not in test_subjs])

    print(f"train list: {train_subjs}")
    print(f"test list: {test_subjs}")

    return train_subjs, test_subjs

# get data for each roi within a parcellation
def get_data_roi(data, cerebellum = 'mdtb_10'):

    """
    get data for the voxels within a roi
    Args:
        data    -   a numpy array with data you want
        cerebellum  -   atlas file name for the parcellation
    Returns:
        data_roi (list)     -   a list containing numpy arrays with the data for all the voxels within each roi
    """

    # print(f"data shape {data.shape}")
    # set the atlas dir
    roiDir = '/home/ladan/Documents/Project/Cerebellum_seq/CerebellarContribution/suit/atlas/atlasesSUIT'

    # get the indices for the each region
    index = cdata.read_suit_nii(os.path.join(roiDir, f'{cerebellum}.nii'))

    region_number_suit = index.astype("int")
    region_numbers = np.unique(region_number_suit)
    num_reg = len(region_numbers)

    # loop over regions and get the data
    data_roi = []
    for r in range(num_reg):
        # get the indices of voxels in suit space
        reg_index = region_number_suit == region_numbers[r]
        # print(reg_index.shape)

        # get data for the region
        reg_data = data[:,reg_index[:, 0]]

        # fill in data_roi
        data_roi.append(reg_data)
    return data_roi

# calculate vip score for a pls model
def calculate_vip(model):

    """
    uses the function here: https://github.com/scikit-learn/scikit-learn/issues/7050
    Args:
        model   -   the model structure

    Returns:
        vips    -   vip scores

    """
    t = model._x_scores
    w = model.x_weights_
    q = model.y_loadings_

    p, _ = w.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips
# Train pls models
def train_pls(
    cortex = 'tessels0162',
    n_components = [2],
    sn = const.return_subjs):

    # train_subjs, _ = split_subjects(sn, test_size=0.3)

    config = run.get_default_train_config()
    num_models = len(n_components)

    for e in range(2):
        # df_all = pd.DataFrame()
        for i in range(num_models):
            name = f"pls_{cortex}_N{n_components[i]}"

            print(f"Doing {name} - {cortex} sc{e+1}")
            config["name"]      = name
            config["model"]     = "PLSRegress"
            config["param"]     = {"n_components": n_components[i]}
            config["X_data"]    = cortex
            config["weighting"] = 2
            config["train_exp"] = f"sc{e+1}"
            config["subjects"]  = sn
            config["mode"] = "crossed"
            config["weighting"] = True
            config["averaging"] = "sess"
            config["validate_model"] = True
            config["cv_fold"] = 4 # other options: 'sess' or 'run' or None
            config["mode"] = "crossed"
            config["hyperparameter"] = f"{n_components[i]:.0f}"
      
            # Model = run.train_models(config, save=True)
            Model = run_connect.train_models(config, save=True)
                   
    return
# eval pls models
def eval_pls(
    cortex = 'tessels0162', 
    n_components = [2], 
    sn=const.return_subjs):

    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(n_components)
    D = pd.DataFrame()
    for i in range(num_models):
        name = f"pls_{cortex}_N{n_components[i]}"
        
        for e in range(2):
            print(f"evaluating {name} - sc{e+1}")
            config["name"] = name
            config["model"] = "PLSRegress"
            config["n_components"] = n_components[i]  # For recording in
            config["X_data"] = cortex
            config["weighting"] = 2
            config["train_exp"] = f'sc{e + 1}'
            config["eval_exp"] = f'sc{2 - e}'
            config["subjects"] = sn
            config["save_maps"] = False
            
            # T = run.eval_models(config)
            T, _ = run_connect.eval_models(config)
            D = pd.concat([D, T], ignore_index=True)

    # check if dataframe already exists
    if os.path.exist(d.conn_eval_dir / f"Pls_{cortex}.dat"):
        DD = pd.read_csv(d.conn_eval_dir / f"Pls_{cortex}.dat") 
        D  = pd.concat([DD, D], ignore_index = True) 
    
    D.to_csv(d.conn_eval_dir / f"Pls_{cortex}.dat")
    return D
# gets vip scores for pls
def get_vips(cortex = 'tessels0162', 
    n_components = [2], 
    sn=const.return_subjs):

    """
    # TO BE IMPLEMENTED:
    calculates variable importance for projection
    1. calculate for each subject in the training dataset: 
            re-do the regression with selected variables on the evaluation dataset
            what are the regions that are important for projection significantly? significantly higher than 1 across subjects
       
    2. do the regression for each cerebellar region (MDTB parcellation, lobules, buckner parcellation)
       calculate vip for each MDTB region
       redo the regression with the selected variables on the evaluation dataset
       For each cerebellar region, what are the cortical regions that are important for projection significantly? significantly higher than 1 across 1 across subjects

    """

    d          = const.Dirs()
    num_models = len(n_components)

    # loop over models
    for e in range(2):
        
        for i in range(num_models):
            name = f"pls_{cortex}_N{n_components[i]}"
            print(f"Calculating VIP for {name} - {cortex} sc{e+1}")
            for s in range(len(sn)):
                print(f"- Getting {sn[s]}")
                subj = sn[s]
                # Get the model from file
                fname = run._get_model_name(name, f"sc{e+1}", subj)
                fitted_model = dd.io.load(fname)

                # calculate vip score
                vip = calculate_vip(fitted_model)

                setattr(fitted_model, "_vip_score", vip)

                fname = run._get_model_name(name, f"sc{e+1}", subj)
                dd.io.save(fname, fitted_model, compression=None)

    return
# create maps for vips for pls
def create_vip_map(cortex = 'tessels0162', 
                   n_components = [2], 
                   train_exp = 'sc1',
                   sn = const.return_subjs):
    """
    creating vip maps for each pls model
    """

    # load in cortical loading and cerebellar loadings maps and create .func.gii
    d = const.Dirs()

    for c in n_components:

        # loop over subjects and get he weights
        name = f"pls_{cortex}_N{c}"
        # get dimensions 
        fname = run._get_model_name(name, train_exp, 'all')
        fitted_model = dd.io.load(fname)
        # get X_lodings
        n = fitted_model._vip_score.shape[0]
        cortex_vip = np.empty([n, len(sn)])
        for s in range(len(sn)):
            print(f"- Getting {sn[s]}")
            subj = sn[s]
            # Get the model from file
            fname = run._get_model_name(name, train_exp, subj)
            fitted_model = dd.io.load(fname)
            # get calculated vip score
            # cortex_vip[:, s] = fitted_model._vip_score
            # get vip scores higher than 1
            select_vip = fitted_model._vip_score >=1

            cortex_vip[:, s] = select_vip
            
            # get functional gifti
            func_gii, hem_names = cdata.convert_cortex_to_gifti(data=cortex_vip, atlas=cortex)

            hemis = ['L', 'R']

            for i, hem in enumerate(hemis):
                fpath = os.path.join('/home/ladan/Desktop/MDTB', f"pls_vip_{cortex}_{c}_{sn[s]}.{hem}.func.gii")
                nib.save(func_gii[i], fpath )
# training pls to predict data for each cerebellar roi within a parcellation
def train_pls_roi(
    cortex = 'tessels0162',
    cerebellum = 'mdtb_10', 
    n_components = [2],
    experiment = 'sc1',
    sn = const.return_subjs):
    """
    training pls models to make predictions for regions of a parcellation
    """

    config = run.get_default_train_config()
    num_models = len(n_components)

    models = []

    for i in range(num_models):
        for s in sn:
            print(f"subject {s}")
            Ydata = cdata.Dataset(experiment=experiment, roi='cerebellum_suit', subj_id=s)
            Ydata.load()
            Y, T = Ydata.get_data(averaging="sess", weighting=2)
            Xdata = cdata.Dataset(experiment=experiment, roi=cortex,subj_id=s)
            Xdata.load()
            X, T = Xdata.get_data(averaging="sess", weighting=2)

            # get the data for each region and fit the model
            Y_roi = get_data_roi(Y, cerebellum = cerebellum)

            # loop over regions and do the modelling
            for r_index, y in enumerate(Y_roi):

                print(r_index)
                name = f"pls_{cortex}_N{n_components[i]}"

                print(f"Doing {name} - {cortex} {experiment} region: {r_index:02}")
                config["name"]      = name
                config["model"]     = "PLSRegress"
                config["param"]     = {"n_components": n_components[i]}
                config["X_data"]    = cortex
                config["weighting"] = 2
                config["train_exp"] = experiment
                config["subjects"]  = [s]
                config["mode"] = "crossed"
        
                # training the model for a specific ROI
                # Generate new model and put in the list
                newModel = getattr(model, config["model"])(**config["param"])
                models.append(newModel)
                # Fit this model
                models[-1].fit(X, y)

                # Save the fitted model to disk if required
                train_name = f"{name}_{cerebellum}"
                dirs = const.Dirs(exp_name=experiment)
                fname = f"{train_name}_{r_index:02}_{s}.h5"
                fpath = os.path.join(dirs.conn_train_dir, cerebellum, f"{r_index:02}", train_name)

                if not os.path.exists(fpath):
                    print(f"creating {fpath}")
                    os.makedirs(fpath)
                dd.io.save(os.path.join(fpath, fname), models[-1], compression=None)
    return
def eval_pls_roi(
    cortex = 'tessels0162',
    cerebellum = 'mdtb_10', 
    n_components = [2],
    sn = const.return_subjs):
    """
    training pls models to make predictions for regions of a parcellation
    """

    config = run.get_default_train_config()
    num_models = len(n_components)

    D = pd.DataFrame()
    for e in range(2):

        for i in range(num_models):
            for s in sn:
                print(f"subject {s}")
                
                name = f"pls_{cortex}_N{n_components[i]}"

                config["name"]      = name
                config["model"]     = "PLSRegress"
                config["param"]     = {"n_components": n_components[i]}
                config["X_data"]    = cortex
                config["weighting"] = 2
                config["train_exp"] = f'sc{e + 1}'
                config["eval_exp"] = f'sc{2 - e}'
                config["subjects"]  = [s]
                config["mode"] = "crossed"

                Ydata = cdata.Dataset(experiment=f"sc{e+1}", roi='cerebellum_suit', subj_id=s)
                Ydata.load()
                Y, T = Ydata.get_data(averaging="sess", weighting=2)
                Xdata = cdata.Dataset(experiment=f"sc{e+1}", roi=cortex,subj_id=s)
                Xdata.load()
                X, T = Xdata.get_data(averaging="sess", weighting=2)

                # get the data for each region and fit the model
                Y_roi = get_data_roi(Y, cerebellum = cerebellum)

                for r_index, y in enumerate(Y_roi):
                    print(r_index)
                    TT = pd.DataFrame()
                    print(f"Doing {name} - {cortex} sc{e+1} region: {r_index:02}")

                    # Get the model from file
                    train_name = f"{name}_{cerebellum}"
                    dirs = const.Dirs(exp_name=config["train_exp"])
                    fname = f"{train_name}_{r_index:02}_{s}.h5"
                    fpath = os.path.join(dirs.conn_train_dir, cerebellum, f"{r_index:02}", train_name)

                    fitted_model = dd.io.load(os.path.join(fpath, fname))

                    # Save the fitted model to disk if required
                    Ypred = fitted_model.predict(X)
                    if config["mode"] == "crossed":
                        Ypred = np.r_[Ypred[T.sess == 2, :], Ypred[T.sess == 1, :]]

                    # Add the subject number
                    TT.loc[r_index, "SN"]= s
                    TT.loc[r_index, "Region"] = r_index
                    TT.loc[r_index, "n_components"] = n_components[i]

                    # Copy over all scalars or strings to the Data frame:
                    for key, value in config.items():
                        if type(value) is not list:
                            TT.loc[r_index, key] = value

                    # Add the evaluation
                    TT.loc[r_index, "R"], Rvox = ev.calculate_R(y, Ypred)  # R between predicted and observed
                    TT.loc[r_index, "R2"], R2vox = ev.calculate_R2(y, Ypred)  # R2 between predicted and observed
                    TT.loc[r_index, "noise_Y_R"], _, TT["noise_Y_R2"], _ = ev.calculate_reliability(y, T)  # Noise ceiling for cerebellum (squared)
                    TT.loc[r_index, "noise_X_R"], _, TT["noise_X_R2"], _ = ev.calculate_reliability(Ypred, T)  # Noise ceiling for cortex (squared)

                    D = pd.concat([D, TT], ignore_index=True)
    return D
# get vip scores for each roi within a parcellation
def get_vips_roi():
    pass    

# train ridge models
def train_ridge(
    cortex = 'tessels0162',
    logalpha = [-2],
    sn = const.return_subjs
    ):
    config = run.get_default_train_config()
    num_models = len(logalpha)
    for i in range(num_models):
        name = f"ridge_{cortex}_A{logalpha[i]:.0f}"
        for e in range(2):
            print(f"Doing {name} - {cortex} sc{e+1}")
            config["name"] = name
            config["param"] = {"alpha": np.exp(logalpha[i])}
            config["X_data"] = cortex
            config["weighting"] = 2
            config["train_exp"] = f"sc{e+1}"
            config["subjects"] = sn
            config["model"]     = "Ridge"
            config["mode"] = "crossed"
            config["weighting"] = True
            config["averaging"] = "sess"
            config["validate_model"] = True
            config["cv_fold"] = 4 # other options: 'sess' or 'run' or None
            config["mode"] = "crossed"
            config["hyperparameter"] = f"{logalpha[i]:.0f}"
            # Model = run.train_models(config, save=True)
            Model = run_connect.train_models(config, save=True)
# eval ridge models    
def eval_ridge(cortex = 'tessels0162', 
    logalpha = [-2], 
    sn=const.return_subjs):

    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(logalpha)
    D = pd.DataFrame()
    for i in range(num_models):
        name = f"ridge_{cortex}_A{logalpha[i]:.0f}"
        for e in range(2):
            print(f"evaluating {name} - sc{e+1}")
            config["name"] = name
            config["logalpha"] = logalpha[i]  # For recording in
            config["X_data"] = cortex
            config["weighting"] = 2
            config["train_exp"] = f'sc{e + 1}'
            config["eval_exp"] = f'sc{2 - e}'
            config["subjects"] = sn
            config["save_maps"] = False
            # T = run.eval_models(config)
            T, _ = run_connect.eval_models(config)

            D = pd.concat([D, T], ignore_index=True)

    # check if dataframe already exists
    if os.path.exist(d.conn_eval_dir / f"Ridge_{cortex}.dat"):
        DD = pd.read_csv(d.conn_eval_dir / f"Ridge_{cortex}.dat") 
        D  = pd.concat([DD, D], ignore_index = True) 
    
    D.to_csv(d.conn_eval_dir / f"Ridge_{cortex}.dat")
    return D

# train lasso models
def train_lasso(
    cortex = 'tessels0162',
    logalpha = [-2],
    sn = const.return_subjs
    ):
    config = run.get_default_train_config()
    num_models = len(logalpha)
    for i in range(num_models):
        name = f"lasso_{cortex}_A{logalpha[i]:.0f}"
        for e in range(2):
            print(f"Doing {name} - {cortex} sc{e+1}")
            config["name"] = name
            config["model"] = "LASSO"
            config["param"] = {"alpha": np.exp(logalpha[i])}
            config["X_data"] = cortex
            config["weighting"] = 2
            config["train_exp"] = f"sc{e+1}"
            config["subjects"] = sn
            config["mode"] = "crossed"
            config["weighting"] = True
            config["averaging"] = "sess"
            config["validate_model"] = True
            config["cv_fold"] = 4 # other options: 'sess' or 'run' or None
            config["mode"] = "crossed"
            config["hyperparameter"] = f"{logalpha[i]:.0f}"
            # Model = run.train_models(config, save=True)
            Model = run_connect.train_models(config, save=True)
# eval lasso models
def eval_lasso(cortex = 'tessels0162', 
    logalpha = [-2], 
    sn=const.return_subjs):

    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(logalpha)
    D = pd.DataFrame()
    for i in range(num_models):
        name = f"lasso_{cortex}_A{logalpha[i]:.0f}"
        for e in range(2):
            print(f"evaluating {name} - sc{e+1}")
            config["name"] = name
            config["model"] = "LASSO"
            config["logalpha"] = logalpha[i]  # For recording in
            config["X_data"] = cortex
            config["weighting"] = 2
            config["train_exp"] = f'sc{e + 1}'
            config["eval_exp"] = f'sc{2 - e}'
            config["subjects"] = sn
            config["save_maps"] = False
            T = run_connect.eval_models(config)
            D = pd.concat([D, T], ignore_index=True)

    # check if dataframe already exists
    if os.path.exist(d.conn_eval_dir / f"Lasso_{cortex}.dat"):
        DD = pd.read_csv(d.conn_eval_dir / f"Lasso_{cortex}.dat") 
        D  = pd.concat([DD, D], ignore_index = True) 
    
    D.to_csv(d.conn_eval_dir / f"Lasso_{cortex}.dat")
    return D

# train wnta models
def train_wnta(cortex = 'tessels0162', 
    n = [2], 
    logalpha = [-2],
    sn=const.return_subjs):

    config = run.get_default_train_config()
    num_models = len(n)
    for i in range(num_models):
        name = f"wnta_{cortex}_N{n[i]:.0f}"
        for e in range(2):
            print(f"Doing {name} - {cortex} sc{e+1}")
            config["name"] = name
            config["model"] = "WNTA"
            config["param"] = {"n": n[i], "alpha":np.exp(logalpha[i])}
            config["X_data"] = cortex
            config["weighting"] = 2
            config["train_exp"] = f"sc{e+1}"
            config["subjects"] = sn
            config["weighting"] = True
            config["averaging"] = "sess"
            # config["validate_model"] = True
            config["validate_model"] = False # no need to validate the model?!
            config["cv_fold"] = 4 # other options: 'sess' or 'run' or None
            config["mode"] = "crossed"
            config["hyperparameter"] = f"{n[i]:.0f}"
            # Model = run.train_models(config, save=True)
            Model = run_connect.train_models(config, save=True)
# eval wnta models
def eval_wnta(cortex = 'tessels0162', 
    n = [1], 
    sn=const.return_subjs):
    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(n)
    D = pd.DataFrame()
    for i in range(num_models):
        name = f"wnta_{cortex}_N{n[i]:.0f}"
        for e in range(2):
            print(f"evaluating {name} - sc{e+1}")
            config["name"] = name
            config["model"] = "WNTA"
            config["n"] = n[i]  # For recording in
            config["X_data"] = cortex
            config["weighting"] = 2
            config["train_exp"] = f'sc{e + 1}'
            config["eval_exp"] = f'sc{2 - e}'
            config["subjects"] = sn
            config["weighting"] = True
            config["averaging"] = "sess"
            config["save_maps"] = False
   
            # T = run.eval_models(config)
            T = run_connect.eval_models(config)
            D = pd.concat([D, T], ignore_index=True)

    # check if dataframe already exists
    if os.path.exist(d.conn_eval_dir / f"Wnta_{cortex}.dat"):
        DD = pd.read_csv(d.conn_eval_dir / f"Wnta_{cortex}.dat") 
        D  = pd.concat([DD, D], ignore_index = True) 
    
    D.to_csv(d.conn_eval_dir / f"Wnta_{cortex}.dat")
    return D

# calculate average weight across group for one component
def calc_average_w_pls(
    cortex = 'tessels0162', 
    model = 'pls',
    param = [9], 
    train_exp = 'sc1',
    sn=const.return_subjs):

    d = const.Dirs()
    # config = run.get_default_eval_config()
    num_models = len(param)
    D = pd.DataFrame()

    # loop over subjects and get he weights
    
    for i in range(num_models):
        name = f"{model}_{cortex}_N{param[i]}"
        w_subs = np.empty([304, 6937, len(sn)])
        for s in range(len(sn)):
            print(f"- Getting {sn[s]}")
            subj = sn[s]
            # Get the model from file
            fname = run._get_model_name(name, train_exp, subj)
            fitted_model = dd.io.load(fname)
            w_subs[:, :, s] = fitted_model.coef_

        w_group = np.nanmean(w_subs, axis = 2)

    return w_group
def calc_average_w_ridge(
    cortex = 'tessels0162', 
    model = 'ridge',
    param = [8], 
    train_exp = 'sc1',
    sn=const.return_subjs):

    d = const.Dirs()
    # config = run.get_default_eval_config()
    num_models = len(param)
    D = pd.DataFrame()

    # loop over subjects and get the weights
    for i in range(num_models):
        name = f"{model}_{cortex}_A{param[i]}"
        # use one subject to initialize arrays
        model_sub = sn[0]
        # Get the model from file
        fname_mod = run._get_model_name(name, train_exp, model_sub)
        fitted_model = dd.io.load(fname_mod)
        model_W = fitted_model.coef_
        print(model_W.shape)

        w_subs = np.empty([model_W.shape[0], model_W.shape[1], len(sn)])
        for s in range(len(sn)):
            print(f"- Getting {sn[s]}")
            subj = sn[s]
            # Get the model from file
            fname = run._get_model_name(name, train_exp, subj)
            fitted_model = dd.io.load(fname)
            w_subs[:, :, s] = fitted_model.coef_

        w_group = np.nanmean(w_subs, axis = 2)

    return w_group
# get the indices for region file in suit
def calc_w_reg_avg(
    W, 
    suit_atlas_name = 'MDTB_10Regions',
    ):
    atlas_dir = '/home/ladan/Documents/Project/Cerebellum_seq/CerebellarContribution/suit/atlas/atlasesSUIT'
    atlas_file = os.path.join(atlas_dir, f"{suit_atlas_name}.nii")
    indx_number = cdata.read_suit_nii(atlas_file)
    indx_number = np.rint(indx_number)
    indx_number = indx_number.astype(int)
    print(indx_number)

    # get the indices for the region
    # loop over regions and get average weights for each
    regions = np.unique(indx_number)
    # print(regions)
    regions = regions[1:]
    # print(regions)
    regions = np.rint(regions).astype(int)
    print(regions)

    W_dict = {}
    for region_number in range(len(regions)):
        print(regions[region_number])
        indx_region = indx_number.T == regions[region_number]
        indx_region = indx_region[0]
        # print(indx_region)

        # get the weights for the region
        W_region = W[:, indx_region]
        # print(W_region)

        # calculate average weight
        W_region_avg = np.nanmean(W_region, axis = 1)

        W_dict[f"region_{region_number}"] = W_region_avg
    
    # convert to dataframe
    W_df = pd.DataFrame(W_dict)

    # save the dataframe
    path = '/home/ladan/Documents/Project/Cerebellum_seq/CerebellarContribution/mdtb_weights/pls'
    name = f"pls_{'tessels162'}_N{7}_{suit_atlas_name}"

    filepath = os.path.join(path, f"{name}.dat")

    W_df.to_csv(filepath)
    return W_region_avg

def create_cortex_maps(
    cortex = 'tessels1442', 
    n_components = 9, 
    train_exp = 'sc1',
    map = 'loadings',
    sn=const.return_subjs
    ):
    # load in cortical loading and cerebellar loadings maps and create .func.gii
    d = const.Dirs()

    # loop over subjects and get he weights
    
    name = f"pls_{cortex}_N{n_components}"
    # get dimensions 
    fname = run._get_model_name(name, train_exp, 's02')
    fitted_model = dd.io.load(fname)
    # get X_lodings
    # n = fitted_model.x_loadings_.shape[0]
    n = fitted_model.x_rotations_ .shape[0]
    print(n)
    cortex_loadings = np.empty([n, n_components, len(sn)])
    for s in range(len(sn)):
        print(f"- Getting {sn[s]}")
        subj = sn[s]
        # Get the model from file
        fname = run._get_model_name(name, train_exp, subj)
        fitted_model = dd.io.load(fname)
        # get X_lodings
        cortex_loadings[:, :, s] = fitted_model.x_rotations_
        # cortex_loadings[:, :, s] = fitted_model.x_weights_ 
        
    group_avg = np.mean(cortex_loadings, axis = 2)

    print(group_avg.shape)

    # get functional gifti
    for i in range(group_avg.shape[1]):
        func_giis, hem_names = cdata.convert_cortex_to_gifti(data=group_avg[:, i], atlas=cortex)

        # save giftis to file
        for (func_gii, hem) in zip(func_giis, hem_names):
            fpath = os.path.join('/home/ladan/Desktop/MDTB', f"mdtb_pls_rotations_{cortex}_N9_{i+1:02d}.{hem}.func.gii")
            nib.save(func_gii, fpath )
    # func_giis, hem_names = cdata.convert_cortex_to_gifti(data=group_avg, atlas=cortex)

    # # save giftis to file
    # for (func_gii, hem) in zip(func_giis, hem_names):
    #     fpath = os.path.join('/home/ladan/Desktop/MDTB', f"mdtb_pls_rotations_{cortex}_N9.{hem}.func.gii")
    #     nib.save(func_gii, fpath )

    return

def create_cerebellum_maps(
    cortex = 'tessels0162', 
    n_components = 9, 
    train_exp = 'sc1',
    sn=const.return_subjs
    ):
    # load in cortical loading and cerebellar loadings maps and create .func.gii
    d = const.Dirs()

    # loop over subjects and get he weights
    
    name = f"pls_{cortex}_N{n_components}"
    cerebellum_loadings = np.empty([6937, 9, len(sn)])
    for s in range(len(sn)):
        print(f"- Getting {sn[s]}")
        subj = sn[s]
        # Get the model from file
        fname = run._get_model_name(name, train_exp, subj)
        fitted_model = dd.io.load(fname)
        # get Y_lodings
        cerebellum_loadings[:, :, s] = fitted_model.y_rotations_
        # cerebellum_loadings[:, :, s] = fitted_model.y_weights_ 
        # print(dir(fitted_model))
        # print(f"scores")
        # print(fitted_model.y_scores_.shape)
        # print(f"weights")
        # print(fitted_model.y_weights_.shape)
        # print(f"scores")
        # print(fitted_model.x_scores_.shape)
        # print(f"weights")
        # print(fitted_model.x_weights_.shape)
        
    group_avg = np.mean(cerebellum_loadings, axis = 2)

    # get functional gifti
    for i in range(group_avg.shape[1]):
        # func_giis, hem_names = cdata.convert_cortex_to_gifti(data=group_avg[:, i], atlas=cortex)
        # convert averaged cerebellum data array to nifti
        nib_obj = cdata.convert_cerebellum_to_nifti(data=group_avg[:, i])[0]
        # map volume to surface
        surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT", stats='nanmean')

        gii_img = flatmap.make_func_gifti(data=surf_data)

        fpath = os.path.join('/home/ladan/Desktop/MDTB', f"mdtb_pls_rotations_{cortex}_N9_{i+1:02d}.cereb.func.gii")
        nib.save(gii_img, fpath)

    return


# Clustering cerebellar voxels
def cluster_w(n_cluster = 9):
    """

    """
    cortex = 'tessels1442'
    experiment = 'sc1'
    sn = 'all'
    logalpha = 8

    name = f"ridge_{cortex}_A{logalpha}"
    # get dimensions 
    fname = run._get_model_name(name, experiment, sn)
    fitted_model = dd.io.load(fname)

    W = fitted_model.coef_

    print(f"doing {n_cluster}")

    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    kk = kmeans.fit(W)
    nib_obj = cdata.convert_cerebellum_to_nifti(data=kk.labels_)[0]
    surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT", stats='mode')
    gii_img = flatmap.make_label_gifti(data=surf_data)
    fpath = os.path.join('/home/ladan/Desktop/MDTB', f"mdtb_ridge_A{logalpha}_cluster_{n_cluster}.cereb.label.gii")
    nib.save(gii_img, fpath)

    return



#
def check_scores(cortex = 'tessels1442', n_components = 8, experiment = 'sc1', sn = 'all'):
    """
    just checking PLS calculated scores for one model
    What are the attributes of a PLSRegress model class?
    # cortical tessels = 2654
    # PLS components = 9
    # conditions = (46 + 46)
    # X (92*2654) matrix of cortical activity profiles
    # Y (92*6937) matrix of cerebellar activity profiles
    x_weights_  : (2654*9)the first right singular vectors of covariance matrix X.T@Y
    y_weights_  : (6937*9)the first left singular vectors of covariance matrix X.T@Y
    ** By definition, weights are chosen so that they maximize the covariance between projected X and the projected Y
       that is Cov(X@x_weights_, Y@y_weights_). 
       In other words, x_weights_ and y_weights_ are the first right and left singular 
       vectors of the covariance matrix.
    _x_scores   : (92, 9)Projections of X on the singular vectors (x_weights_)
    _y_scores   : (92, 9)Projections of Y on the singular vectors (y_weights_)
    x_loadings_ : (2654*9)regress X on _x_scores to obtain x_loadings
    y_loadings_ : (6937*9)regress Y on _y_scores to obrain y_loadings
    x_rotations_: (2654*9)Projection matrix that takes X and project it to the latent space
    y_rotations_: (6937*9)Projection matrix that takes Y and project it to the latent space
    """
    name = f"pls_{cortex}_N{n_components}"

    fname = run._get_model_name(name, experiment, sn)
    model = dd.io.load(fname)
    x = model._x_scores
    y = model._y_scores

    # relationship between scores
    sb.scatterplot(x[:, 0], y[:, 0])
    sb.scatterplot(x[:, 0], y[:, 1])

    # dot product of scores
    ## different components will be 0
    dot_prod = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            dot_prod[i, j] = np.dot(x[:, i], y[:, j])

    plt.imshow(dot_prod)
    plt.colorbar()

    # orthonormal loadings?
    ## xx yes, yy no
    xx = model.x_loadings_
    yy = model.y_loadings_

    aa = (xx.T)@(xx)
    plt.imshow(aa)

    bb = (yy.T)@(yy)
    plt.imshow(bb)

    # plotting latent variables of X against each other
    sb.scatterplot(x[:, 0], x[:, 1])

def perform_permutation(cortex = 'tessels1442', n_components = 8, experiment = 'sc1', sn = 'all'):
    """
    performing permutations to identify generalizable saliences as in abdi_2010
    """
    num_iter = 100

    # fit the original model
    name = f"pls_{cortex}_N{n_components}"

    config = run.get_default_train_config()

    print(f"Doing {name} - {cortex} {experiment}")
    config["name"]      = name
    config["model"]     = "PLSRegress"
    config["param"]     = {"n_components": n_components}
    config["X_data"]    = cortex
    config["weighting"] = 2
    config["train_exp"] = f"{experiment}"
    config["subjects"]  = sn
    config["mode"] = "crossed"

    Model = run.train_models(config, save=False)

    # getting ready for permutations
    # load in data
    # Get the condensed data
    Ydata = Dataset(glm=config["glm"], subj_id=sn, roi=config["Y_data"])
    Ydata.load()
    Y, T = Ydata.get_data(averaging=config["averaging"], weighting=config["weighting"])
    Xdata = Dataset(glm=config["glm"], subj_id=sn, roi=config["X_data"])
    Xdata.load()
    X, T = Xdata.get_data(averaging=config["averaging"], weighting=config["weighting"])

    # permuting
    iter = 0
    while iter<num_iter:
        # create a copy of X
        X_copy = copy.deepcopy(X)

        # randomly shuffle rows of X_copy
        np.random.shuffle(X_copy)

        # fit the model again with the shuffled data 
        newModel = getattr(model, config["model"])(**config["param"])

def CV_pls(sn = const.return_subjs, 
           cortex = 'tessels0162', train_exp = 'sc1'):
    train_subjs, test_subjs = split_subjects(sn, test_size=0.3)

    # creating the group average
    Y = Dataset(experiment = train_exp, roi = 'cerebellum_suit', subj_id = train_subjs) # Any list of subjects will do (experiment=experiment, roi='cerebellum_suit', subj_id=s)
    Y.load_mat()                             # Load from Matlab
    Y.average_subj()                         # Average 

    X = Dataset(experiment = train_exp, roi = cortex, subj_id = train_subjs) # Any list of subjects will do (experiment=experiment, roi='cerebellum_suit', subj_id=s)
    X.load_mat()                             # Load from Matlab
    X.average_subj()                         # Average 


    print(Y.shape)
    print(X.shape)

    pass

def pipeline():

    try:
        train_wnta(cortex = 'gordon', n = [2, 3, 4, 5, 6, 7])
        train_wnta(cortex = 'fan', n = [2, 3, 4, 5, 6, 7])
        train_wnta(corte = 'shen', n = [2, 3, 4, 5, 6, 7])
        eval_wnta(cortex = 'glasser', n = [2, 3, 4, 5, 6, 7])
        eval_wnta(cortex = 'fan', n = [2, 3, 4, 5, 6, 7])
        eval_wnta(cortex = 'shen', n = [2, 3, 4, 5, 6, 7])
        eval_wnta(cortex = 'baldassano', n = [2, 3, 4, 5, 6, 7])
        eval_wnta(cortex = 'gordon', n = [2, 3, 4, 5, 6, 7])
    except:
        print('encountered error')
    return
if __name__ == "__main__":
    
    # estimating models
    D1 = train_pls(cortex = 'tessels0162', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])
    D2 = train_pls(cortex = 'tessels0362', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])
    D3 = train_pls(cortex = 'tessels0642', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])
    D3 = train_pls(cortex = 'tessels1002', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])

    # evaluating models
    D3 = eval_pls(cortex = 'tessels0162', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])
    D3 = eval_pls(cortex = 'tessels0362', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])
    D3 = eval_pls(cortex = 'tessels0642', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])
    D3 = eval_pls(cortex = 'tessels1002', n_components=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20])


