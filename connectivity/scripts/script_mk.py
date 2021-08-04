# import libraries
import os, shutil
import click
import numpy as np
import pandas as pd
import nibabel as nib
import glob
from scipy.stats import mode
from random import seed, sample
import neptune
from pathlib import Path
import SUITPy.flatmap as flatmap

import connectivity.constants as const
import connectivity.io as cio
from connectivity import data as cdata
import connectivity.run_mk as run_connect
from connectivity import visualize as summary

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
    experimenter='mk'
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
        experimenter (str or None): 'mk' or 'ls' or None
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
    if experimenter:
        fpath = os.path.join(dirs.conn_train_dir, f'train_summary_{experimenter}.csv')
    else:
        fpath = os.path.join(dirs.conn_train_dir, f'train_summary.csv')

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
    experimenter='mk'
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
        experimenter (str or None): 'mk' 'sh' etc.
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
    if experimenter:
        fpath = os.path.join(dirs.conn_train_dir, f'train_summary_{experimenter}.csv')
    else:
        fpath = os.path.join(dirs.conn_train_dir, f'train_summary.csv')

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
    experimenter='mk'
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
        experimenter (str or None): 'mk' or 'ls' or None
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
    if experimenter:
        fpath = os.path.join(dirs.conn_train_dir, f'train_summary_{experimenter}.csv')
    else:
        fpath = os.path.join(dirs.conn_train_dir, f'train_summary.csv')

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

    # save maps to disk for cortex
    data = np.stack(cortex_weights_all, axis=0)
    func_giis, hem_names = cdata.convert_cortex_to_gifti(data=np.nanmean(data, axis=0), atlas=cortex)
    for (func_gii, hem) in zip(func_giis, hem_names):
        nib.save(func_gii, os.path.join(fpath, f'group_weights_cortex.{hem}.func.gii'))

    print('saving cortical and cerebellar weights to disk')

def save_maps_cerebellum(
    data, 
    fpath='/',
    group='nanmean', 
    gifti=True, 
    nifti=True, 
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

def save_lasso_maps(
    model_name, 
    train_exp, 
    stat='count'
    ):
    """save lasso maps for cerebellum (count number of non-zero cortical coef)

    Args:
        model_name (str): full name of trained model
        train_exp (str): 'sc1' or 'sc2'
        stat (str): 'count' or 'percent'
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    cereb_lasso_all = []
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)
        
        # count number of non-zero weights
        data_nonzero = np.count_nonzero(data.coef_, axis=1)

        if stat=='count':
            pass # do nothing
        elif stat=='percent':
            num_regs = data.coef_.shape[1]
            data_nonzero = np.divide(data_nonzero,  num_regs)*100
        cereb_lasso_all.append(data_nonzero)

    # save maps to disk for cerebellum
    save_maps_cerebellum(data=np.stack(cereb_lasso_all, axis=0), 
                        fpath=os.path.join(fpath, f'group_lasso_{stat}_cerebellum'))

def eval_model(
    model_name,
    train_exp="sc1",
    eval_exp="sc2",
    cortex="tesselsWB642",
    cerebellum="cerebellum_suit",
    log_online=False,
    log_locally=True,
    experimenter='mk'
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
        experimenter (str or None): 'mk' or 'ls' or None
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

    # eval summary
    if experimenter:
        eval_fpath = os.path.join(dirs.conn_eval_dir, f'eval_summary_{experimenter}.csv')
    else:
        eval_fpath = os.path.join(dirs.conn_eval_dir, f'eval_summary.csv')

    # concat data to eval summary (if file already exists)
    if log_locally:
        if os.path.isfile(eval_fpath):
            df = pd.concat([df, pd.read_csv(eval_fpath)])
        df.to_csv(eval_fpath, index=False)

def _delete_models(exp, best_model):
    dirs = const.Dirs(exp_name=exp)
    model_fpaths = [f.path for f in os.scandir(dirs.conn_train_dir) if f.is_dir()]
    for fpath in model_fpaths:
        if best_model != Path(fpath).name:
            shutil.rmtree(fpath)
        
def _log_models(exp):
    dirs = const.Dirs(exp_name=exp)

    dataframe = summary.train_summary(exps=[exp])

    # groupby train_name
    dataframe = dataframe.groupby('name').first().reset_index()[['train_name', 'train_exp', 'train_X_data', 'train_Y_data', 'train_model', 'train_glm', 'train_averaging', 'train_validate_model', 'train_weighting']]
    
    fpath = os.path.join(dirs.base_dir, 'model_logs.csv')
    if os.path.isfile(fpath):
        dataframe = pd.concat([dataframe, pd.read_csv(fpath)])
   
    # save out train summary
    dataframe.to_csv(fpath, index=False)

def _check_eval(model_name, train_exp, eval_exp):
    """determine if `model_name` should be evaluated

    Args: 
        model_name (str): 
        train_exp (str): 
        eval_exp (str):
    Returns: 
        eval (bool)
    """
    
    train_subjs, _ = split_subjects(const.return_subjs, test_size=0.3)

    eval = True
    # check if trained model is complete (all `train_subjs`)
    dirs = const.Dirs(exp_name=train_exp)
    for subj in train_subjs:
        fname = f'{model_name}_{subj}.h5'
        if not os.path.exists(os.path.join(dirs.conn_train_dir, model_name, fname)):
            return False

    # check if trained model has already been evaluted 
    dirs = const.Dirs(exp_name=eval_exp)
    if os.path.isdir(os.path.join(dirs.conn_eval_dir, model_name)):
        eval = False

    return eval

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
            _log_models(exp=f"sc{exp+1}")

    elif train_or_eval=="eval":
        for exp in range(2):
            
            # get best model (for each method and parcellation)
            models, cortex_names = summary.get_best_models(train_exp=f"sc{2-exp}")

            for (best_model, cortex) in zip(models, cortex_names):
                
                # should trained model be evaluated?
                eval = _check_eval(model_name=best_model, train_exp=f"sc{2-exp}", eval_exp=f"sc{exp+1}")

                ### TEMP ###
                if 'lasso' in best_model:
                    save_lasso_maps(model_name=best_model, train_exp=f"sc{2-exp}", stat='percent') 
                
                if eval:
                    # save voxel/vertex maps for best training weights (for group parcellations only)
                    if 'wb_indv' not in cortex:
                        save_weight_maps(model_name=best_model, train_exp=f"sc{2-exp}", stat='count')
                        save_weight_maps(model_name=best_model, train_exp=f"sc{2-exp}", stat='percent')

                    if 'lasso' in best_model:
                        save_lasso_maps(model_name=best_model, train_exp=f"sc{2-exp}")  

                    # delete training models that are suboptimal (save space)
                    if delete_train:
                        _delete_models(exp=f"sc{2-exp}", best_model=best_model)

                    # test best train model
                    eval_model(model_name=best_model, cortex=cortex, train_exp=f"sc{2-exp}", eval_exp=f"sc{exp+1}")
                else:
                    print(f'{best_model} was not evaluated')


if __name__ == "__main__":
    run()
