# import libraries
import os, shutil
import click
import numpy as np
import pandas as pd
import nibabel as nib
import glob
from scipy.stats import mode
from random import seed, sample
from collections import defaultdict
import neptune
from pathlib import Path
import SUITPy.flatmap as flatmap

import connectivity.constants as const
import connectivity.io as cio
import connectivity.nib_utils as nio
from connectivity import data as cdata
import connectivity.run_mk as run_connect
from connectivity import visualize as summary


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


def log_to_neptune(
    dataframe, 
    config, 
    modeltype="train"
    ):
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
            neptune.set_property(k, v)
    neptune.stop()


def train_ridge(
    hyperparameter,
    train_exp="sc1",
    cortex="tessels0642",
    cerebellum="cerebellum_suit",
    log_online=False,
    log_locally=True,
    model_ext=None,
    ):
    """Train model

    Optimal alpha value is returned based on R_cv.

    Args:
        hyperparameter (list): list of alpha values.
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
    for param in hyperparameter:
        print(f"training param {param:.0f}")
        name = f"ridge_{cortex}_alpha_{param:.0f}" # important that model naming convention stays this way!
        if model_ext is not None:
            name = f"{name}_{model_ext}"
        config["name"] = name
        config["param"] = {"alpha": np.exp(param)}
        config["model"] = "L2regression"
        config["X_data"] = cortex
        config["Y_data"] = cerebellum
        config["weighting"] = True
        config["averaging"] = "sess"
        config["train_exp"] = train_exp
        config["subjects"] = train_subjs
        config["validate_model"] = True
        config["cv_fold"] = 4 # other options: 'sess' or 'run' or None
        config["mode"] = "crossed"
        config["hyperparameter"] = f"{param:.0f}"

        # train model
        models, df = run_connect.train_models(config, save=log_locally)
        df_all = pd.concat([df_all, df])

        # write online to neptune
        if log_online:
            log_to_neptune(dataframe=df, config=config, modeltype="train")

    # save out train summary
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.conn_train_dir, "train_summary.csv")

    # save out weight maps
    if config['save_weights']:
        save_weight_maps(model_name=name, cortex=cortex, train_exp=train_exp)

    # concat data to model_summary (if file already exists)
    if log_locally:
        if os.path.isfile(fpath):
            df_all = pd.concat([df_all, pd.read_csv(fpath)])
        # save out train summary
        df_all.to_csv(fpath, index=False)


def train_WTA(
    train_exp="sc1",
    cortex="tessels0642",
    cerebellum="cerebellum_suit",
    positive=True,
    log_online=False,
    log_locally=True,
    model_ext=None,
    ):
    """Train model

    Optimal alpha value is returned based on R_cv.

    Args:
        train_exp (str): 'sc1' or 'sc2'
        cortex (str): cortical ROI
        cerebellum (str): cerebellar ROI
        positive (bool): if True, take only positive coeficients, if False, take absolute
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
    name = f"WTA_{cortex}"
    if model_ext is not None:
        name = f"{name}_{model_ext}"
    config["name"] = name
    config["param"] = {"positive": positive}
    config["model"] = 'WTA'
    config["X_data"] = cortex
    config["Y_data"] = cerebellum
    config["weighting"] = True
    config["averaging"] = "sess"
    config["train_exp"] = train_exp
    config["subjects"] = train_subjs
    config["validate_model"] = True
    config["cv_fold"] = 4
    config["mode"] = "crossed"
    config["hyperparameter"] = 0

    # train model
    models, df = run_connect.train_models(config, save=log_locally)
    df_all = pd.concat([df_all, df])

    # write online to neptune
    if log_online:
        log_to_neptune(dataframe=df, config=config, modeltype="train")

    # save out train summary
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.conn_train_dir, "train_summary.csv")

    # save out weight maps
    if config['save_weights']:
        save_weight_maps(model_name=name, cortex=cortex, train_exp=train_exp)

    # concat data to model_summary (if file already exists)
    if log_locally:
        if os.path.isfile(fpath):
            df_all = pd.concat([df_all, pd.read_csv(fpath)])
        # save out train summary
        df_all.to_csv(fpath, index=False)


def train_NNLS(
    alphas,
    gammas,
    train_exp="sc1",
    cortex="tessels0642",
    cerebellum="cerebellum_suit",
    log_online=False,
    log_locally=True,
    model_ext=None,
    ):
    """Train model

    Optimal alpha value is returned based on R_cv.

    Args:
        hyperparameter (list): list of alpha values.
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
    for (alpha, gamma) in zip(alphas, gammas):
        print(f"training alpha {alpha:.0f} training gamm {gamma:.0f}")
        name = f"NNLS_{cortex}_alpha_{alpha:.0f}_gamma_{gamma:.0f}" # important that model naming convention stays this way!
        if model_ext is not None:
            name = f"{name}_{model_ext}"
        config["name"] = name
        config["param"] = {"alpha": np.exp(alpha), "gamma": gamma}
        config["model"] = "NNLS"
        config["X_data"] = cortex
        config["Y_data"] = cerebellum
        config["weighting"] = True
        config["averaging"] = "sess"
        config["train_exp"] = train_exp
        config["subjects"] = train_subjs
        config["validate_model"] = False
        config["cv_fold"] = 4
        config["mode"] = "crossed"
        config["hyperparameter"] = f"{alpha:.0f}_{gamma:.0f}"

        # train model
        models, df = run_connect.train_models(config, save=log_locally)
        df_all = pd.concat([df_all, df])

        # write online to neptune
        if log_online:
            log_to_neptune(dataframe=df, config=config, modeltype="train")

    # save out train summary
    dirs = const.Dirs(exp_name=train_exp)
    fpath = os.path.join(dirs.conn_train_dir, "train_summary.csv")

    # save out weight maps
    if config['save_weights']:
        save_weight_maps(model_name=name, cortex=cortex, train_exp=train_exp)

    # concat data to model_summary (if file already exists)
    if log_locally:
        if os.path.isfile(fpath):
            df_all = pd.concat([df_all, pd.read_csv(fpath)])
        # save out train summary
        df_all.to_csv(fpath, index=False)


def save_weight_maps(
        model_name, 
        cortex, 
        train_exp
        ):
    """Save weight maps to disk for cortex and cerebellum

    Args: 
        model_name (str): model_name (folder in conn_train_dir)
        cortex (str): cortex model name (example: tesselsWB162)
        train_exp (str): 'sc1' or 'sc2'
    Returns: 
        saves nifti/gifti to disk
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    cereb_weights_all = []; cortex_weights_all = []
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)
        
        # append cerebellar and cortical weights
        cereb_weights_all.append(np.nanmean(data.coef_, axis=1))
        cortex_weights_all.append(np.nanmean(data.coef_, axis=0))

    # save maps to disk for cerebellum and cortex
    save_maps_cerebellum(data=np.stack(cereb_weights_all, axis=0), 
                        fpath=os.path.join(fpath, 'group_weights_cerebellum'))

    save_maps_cortex(data=np.stack(cortex_weights_all, axis=0),
                    atlas=cortex,
                    fpath=os.path.join(fpath, 'group_weights_cortex'))

    print('saving cortical and cerebellar weights to disk')


def save_wta_maps(
        model_name, 
        cortex, 
        train_exp
        ):
    """Save weight maps to disk for cortex and cerebellum

    Args: 
        model_name (str): model_name (folder in conn_train_dir)
        cortex (str): cortex model name (example: tesselsWB162)
        train_exp (str): 'sc1' or 'sc2'
    Returns: 
        saves nifti/gifti to disk
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    labels_all = []
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)
        
        # append labels
        labels_all.append(data.labels)

    # save maps to disk for cerebellum and cortex
    save_maps_cerebellum(data=np.stack(labels_all, axis=0), 
                        fpath=os.path.join(fpath, 'group_wta_cerebellum'),
                        group='mode',
                        nifti=True)


def save_maps_cerebellum(
    data, 
    fpath='/',
    group='nanmean', 
    gifti=True, 
    nifti=False, 
    column_names=[], 
    label_RGBA=[],
    label_names=[],
    ):
    """Takes data (np array), averages along first dimension
    saves nifti and gifti map to disk

    Args: 
        data (np array): np array of shape (N x 6937)
        fpath (str): save path for output file
        group (bool): default is 'nanmean' (for func data), other option is 'mode' (for label data) 
        gifti (bool): default is True, saves gifti to fpath
        nifti (bool): default is False, saves nifti to fpath
        column_names (list):
        label_RGBA (list):
        label_names (list):
    Returns: 
        saves nifti and/or gifti image to disk, returns gifti
    """
    num_cols, num_vox = data.shape

    # get mean or mode of data along first dim (first dim is usually subjects)
    if group=='nanmean':
        data = np.nanmean(data, axis=0)
    elif group=='mode':
        data = mode(data, axis=0)
        data = data.mode[0]
    else:
        print('need to group data by passing "nanmean" or "mode"')

    # convert averaged cerebellum data array to nifti
    nib_obj = cdata.convert_cerebellum_to_nifti(data=data)[0]
    
    # save nifti(s) to disk
    if nifti:
        nib.save(nib_obj, fpath + '.nii')

    # map volume to surface
    surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT", stats=group)

    # make and save gifti image
    if group=='nanmean':
        gii_img = flatmap.make_func_gifti(data=surf_data, column_names=column_names)
        out_name = 'func'
    elif group=='mode':
        gii_img = flatmap.make_label_gifti(data=surf_data, label_names=label_names, column_names=column_names, label_RGBA=label_RGBA)
        out_name = 'label'
    if gifti:
        nib.save(gii_img, fpath + f'.{out_name}.gii')
    
    return gii_img


def save_maps_cortex(
    data, 
    atlas, 
    fpath='/', 
    group_average=True
    ):
    """Takes list of np arrays, averages list and
    saves gifti map to disk

    Args: 
        data (np array): np array of shape (N x 32492)
        fpath (str): save path for output file
        atlas (str): cortex atlas name (example: tessels0162)
        group_average (bool): default is True, averages data np arrays 
    Returns: 
        saves gifti image to disk for left and right hemispheres
    """
    # average data
    if group_average:
        data = np.nanmean(data, axis=0)

    # get functional gifti
    func_giis, hem_names = cdata.convert_cortex_to_gifti(data=data, atlas=atlas)
    
    # save giftis to file
    for (func_gii, hem) in zip(func_giis, hem_names):
        nib.save(func_gii, fpath + f'.{hem}.func.gii')


def eval_model(
    model_name,
    train_exp="sc1",
    eval_exp="sc2",
    cortex="tesselsWB642",
    cerebellum="cerebellum_suit",
    log_online=False,
    log_locally=True,
    ):
    """Evaluate model(s)

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

    # save voxel data to gifti(only for cerebellum_suit)
    if config["save_maps"] and config["Y_data"] == "cerebellum_suit":
        fpath = os.path.join(dirs.conn_eval_dir, model_name)
        cio.make_dirs(fpath)
        for k, v in voxels.items():
            save_maps_cerebellum(data=np.stack(v, axis=0), 
                                fpath=os.path.join(fpath, f'group_{k}'))

    # write to neptune
    if log_online:
        log_to_neptune(dataframe=df, config=config, modeltype="eval")

    # concat data to eval summary (if file already exists)
    if log_locally:
        eval_fpath = os.path.join(dirs.conn_eval_dir, f"eval_summary.csv")
        if os.path.isfile(eval_fpath):
            df = pd.concat([df, pd.read_csv(eval_fpath)])
        df.to_csv(eval_fpath, index=False)
        
def log_models(exp):
    dirs = const.Dirs(exp_name=exp)
    dataframe = pd.read_csv(os.path.join(dirs.conn_train_dir, "train_summary.csv"))

    # groupby train_name
    dataframe = dataframe.groupby('name').first().reset_index()[['name', 'train_exp', 'X_data', 'Y_data', 'model', 'glm', 'averaging', 'validate_model', 'weighting']]
    
    fpath = os.path.join(dirs.base_dir, 'model_logs.csv')
    if os.path.isfile(fpath):
        dataframe = pd.concat([dataframe, pd.read_csv(fpath)])
   
    # save out train summary
    dataframe.to_csv(fpath, index=False)


@click.command()
@click.option("--cortex")
@click.option("--model_type")
@click.option("--train_or_eval")


def run(cortex="tessels0362", 
        model_type="ridge", 
        train_or_eval="train", 
        delete_train=False):
    """ Run connectivity routine (train and evaluate)

    Args: 
        cortex (str): 'tesselsWB162', 'tesselsWB642' etc.
        model_type (str): 'WTA' or 'ridge' or 'NNLS'
        train_or_test (str): 'train' or 'eval'
    """
    print(f'doing model {train_or_eval}')
    if train_or_eval=="train":
        for exp in range(2):
            if model_type=="ridge":
                # train ridge
                train_ridge(hyperparameter=[-2,0,2,4,6,8,10], train_exp=f"sc{exp+1}", cortex=cortex)
            elif model_type=="WTA":
                train_WTA(train_exp=f"sc{exp+1}", cortex=cortex)
            elif model_type=="NNLS":
                train_NNLS(alphas=[0], gammas=[0], train_exp=f"sc{exp+1}", cortex=cortex)
            else:
                print('please enter a model (ridge, WTA, NNLS)')
            
            # log models
            log_models(exp=f"sc{exp+1}")

    elif train_or_eval=="eval":
        for exp in range(2):
            
            # get best model for each method and parcellation (NNLS, ridge, WTA)
            models, cortex_names = summary.get_best_models(train_exp=f"sc{2-exp}")

            # get best train model (based on train CV)
            # best_model, cortex = summary.get_best_model(train_exp=f"sc{2-exp}")

            for (best_model, cortex) in zip(models, cortex_names):

                # save voxel/vertex maps for best training weights
                save_weight_maps(model_name=best_model, cortex=cortex, train_exp=f"sc{2-exp}")

                # delete training models that are suboptimal (save space)
                if delete_train:
                    dirs = const.Dirs(exp_name=f"sc{2-exp}")
                    model_fpaths = [f.path for f in os.scandir(dirs.conn_train_dir) if f.is_dir()]
                    for fpath in model_fpaths:
                        if best_model != Path(fpath).name:
                            shutil.rmtree(fpath)

                # test best train model
                eval_model(model_name=best_model, cortex=cortex, train_exp=f"sc{2-exp}", eval_exp=f"sc{exp+1}")


if __name__ == "__main__":
    run()
