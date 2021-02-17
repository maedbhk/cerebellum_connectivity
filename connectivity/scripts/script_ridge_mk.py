# import libraries
import os, shutil
import click
import numpy as np
import pandas as pd
import glob
from random import seed, sample
from collections import defaultdict
import neptune

import connectivity.constants as const
import connectivity.io as cio
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run_mk as run_connect


def delete_conn_files():
    """delete any pre-existing connectivity output."""
    for exp in ["sc1", "sc2"]:
        dirs = const.Dirs(exp_name=exp, glm="glm7")
        filelists = [
            glob.glob(os.path.join(dirs.conn_train_dir, "*")),
            glob.glob(os.path.join(dirs.conn_eval_dir, "*")),
        ]
        for filelist in filelists:
            for f in filelist:
                try:
                    shutil.rmtree(f)
                except:
                    os.remove(f)
    print("deleting training and evaluation connectivity data")


def split_subjects(subj_ids, test_size=0.3):
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


def log_to_neptune(dataframe, config, modeltype="train"):
    """log training and evaluation data to neptune (ML experiment tracker)
    
    This case won't work unless you have registered with neptune and get your own api_token

    Args: 
        dataframe (pd dataframe): data you want to log
        config (dict): keys you want to log
        modeltype (str): 'train' or 'eval'
    """
    # set up experiment
    print("tracking experiment")
    neptune.init(
        project_qualified_name=f"maedbhking/connectivity",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiY2ExNWUwODktZDBmNS00YzdjLTg3YWUtMWYzNTE1ZmExYWZlIn0=",
    )
    neptune.create_experiment()

    # data cols to log
    cols = [col for col in dataframe.columns if any(s in col for s in ("R", "rmse"))]
    df = dataframe[cols].mean()  # get average value (across subjs)

    # log metrics to neptune
    for idx in df.index:
        neptune.log_metric(idx, df[idx])

    # append model type
    neptune.append_tag(modeltype)

    # add config values as tags
    for k, v in config.items():
        if not isinstance(v, (list, dict)):
            neptune.set_property(k,v)
    neptune.stop()


def train_ridge(log_alpha, train_exp="sc1",  cortex='tesselsWB162', cerebellum='cerebellum_suit', log_online=False, log_locally=True, model_ext=None):
    """Train ridge model(s) using different alpha values.

    Optimal alpha value is returned based on R_cv.

    Args:
        log_alpha (list): list of alpha values.
        train_exp (str): 'sc1' or 'sc2'
        cortex (str): cortical ROI
        cerebellum (str): cerebellar ROI
        log_online (bool): log results to ML tracking platform
        log_locally (bool): log results locally
        model_ext (str or None): add additional information to base model name
    Returns:
        Appends summary data for each model and subject into `train_summary.csv`
        Returns pandas dataframe of train_summary
    """
    train_subjs, _ = split_subjects(const.return_subjs, test_size=0.3)

    # get default train parameters
    config = run_connect.get_default_train_config()

    df_all = pd.DataFrame()
    # train and validate ridge models
    for alpha in log_alpha:
        print(f"training alpha {alpha:.0f}")
        name = f"ridge_{cortex}_alpha_{alpha:.0f}"
        if model_ext is not None:
            name = f"{name}_{model_ext}"
        config["name"] = name
        config["param"] = {"alpha": np.exp(alpha)}
        config["X_data"] = cortex
        config["Y_data"] = cerebellum
        config["weighting"] = True
        config["averaging"] = "sess"
        config["train_exp"] = train_exp
        config["subjects"] = train_subjs
        config["validate_model"] = True
        config["cv_fold"] = 4
        config["mode"] = "crossed"
        config["hyperparameter"] = f"{alpha:.0f}"

        # train model
        models, df = run_connect.train_models(config, save=log_locally)
        df_all = pd.concat([df_all, df])

        # write online to neptune
        if log_online:
            log_to_neptune(dataframe=df, config=config, modeltype="train")

    # save out train summary
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.conn_train_dir, "train_summary.csv")

    # concat data to model_summary (if file already exists)
    if log_locally:
        if os.path.isfile(fpath):
            df_all = pd.concat([df_all, pd.read_csv(fpath)])
        # save out train summary
        df_all.to_csv(fpath, index=False)


def eval_ridge(model_name, train_exp="sc1", eval_exp="sc2", cortex='tesselsWB162', cerebellum='cerebellum_suit', log_online=False, log_locally=True):
    """Evaluate ridge model(s) using different alpha values.

    Args:
        model_name (str): name of trained model
        train_exp (str): 'sc1' or 'sc2'
        eval_exp (str): 'sc1' or 'sc2'
        cortex (str): cortical ROI
        cerebellum (str): cerebellar ROI
        log_online (bool): log results to ML tracking platform
        log_locally (bool): log results locally
        model_ext (str or None): add additional information to base model name.
    Returns:
        Appends eval data for each model and subject into `eval_summary.csv`
        Returns pandas dataframe of eval_summary
    """
    dirs = const.Dirs(exp_name=eval_exp)

    train_subjs, _ = split_subjects(const.return_subjs, test_size=0.3)

    # get default eval parameters
    config = run_connect.get_default_eval_config()

    print(f"evaluating {model_name}")
    config["name"] = model_name
    config["X_data"] = cortex
    config["Y_data"] = cerebellum
    config["weighting"] = True
    config["averaging"] = "sess"
    config["train_exp"] = train_exp
    config["eval_exp"] = eval_exp
    config["subjects"] = train_subjs
    config["save_maps"] = True

    # eval model(s)
    df, voxels = run_connect.eval_models(config)

    # save voxel data (only for cerebellum_suit)
    if config["save_maps"] and config["Y_data"] == "cerebellum_suit":
        fpath = os.path.join(dirs.conn_eval_dir, model_name)
        cio.make_dirs(fpath)
        save_voxels(data=voxels, fpath=os.path.join(fpath, "voxels_group.h5"))

    # write to neptune
    if log_online:
        log_to_neptune(dataframe=df, config=config, modeltype="eval")

    # concat data to eval summary (if file already exists)
    fpath = os.path.join(dirs.conn_eval_dir, f"eval_summary.csv")
    if log_locally:
        if os.path.isfile(fpath): 
            df = pd.concat([df, pd.read_csv(fpath)])
        df.to_csv(fpath, index=False)


def save_voxels(data, fpath):
    """Averages subj-level voxel output from model training/evaluation 
    and saves to disk.

    Args: 
        data (dict): dictionary containing subj-level model predictions (R_vox, R2_vox)
        fpath (str): full path to voxel output
    Returns: 
        saves 1-D arrays (group data) to disk (.h5)
    """
    avg_data = {}
    # loop over keys and average across subjs
    for k, v in data.items():
        avg_data.update({k: np.array([float(sum(col)) / len(col) for col in zip(*v)])})

    # calculate noise ceiling
    avg_data["noise_ceiling_Y"] = np.sqrt(avg_data["noise_Y_R_vox"])
    avg_data["noise_ceiling_XY"] = np.sqrt(avg_data["noise_Y_R_vox"] * np.sqrt(avg_data["noise_X_R_vox"]))

    # save voxels to disk
    cio.save_dict_as_hdf5(fpath=fpath, data_dict=avg_data)


def get_best_model(exp):
    """Get idx for best ridge based on either rmse_train or rmse_cv.

    If rmse_cv is populated, this is used to determine best ridge.
    Otherwise, rmse_train is used.

    Args:
        exp (str): 'sc1' or 'sc2
    Returns:
        model name (str)
    """
    # load train summary (contains R CV of all trained models)
    dirs = const.Dirs(exp_name=exp)
    fpath = os.path.join(dirs.conn_train_dir, "train_summary.csv")
    df = pd.read_csv(fpath)

    # get mean values for each model
    tmp = df.groupby('name').mean().reset_index()

    # get best model (based on R CV)
    best_model = tmp[tmp['R_cv']==tmp['R_cv'].max()]['name'].values[0]

    print(f'best model for {exp} is {best_model}')

    return best_model


def run():
    # train models
    for exp in range(2):
        # train ridge
        train_ridge(log_alpha=[0, 2, 4, 6, 8, 10], train_exp=f"sc{exp+1}")

    # eval models
    for exp in range(2):
        # get best train model (based on train CV)
        model_name = get_best_model(exp=f"sc{2-exp}")

        # test best train model
        eval_ridge(model_name=model_name, train_exp=f"sc{2-exp}", eval_exp=f"sc{exp+1}")


if __name__ == '__main__':
    run()
