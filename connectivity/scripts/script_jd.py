import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import os
import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.data as cdata
import connectivity.run as run
import connectivity.visualize as vis
import connectivity.figures as fig
import connectivity.io as cio
from SUITPy import flatmap
import itertools
import nibabel as nib
import h5py
import deepdish as dd

def train_ridge(corticalParc, logalpha, sn=const.return_subjs):
    config = run.get_default_train_config()
    num_models = len(logalpha)
    for i in range(num_models):
        name = f"ridge_{corticalParc}_A{logalpha[i]:.0f}"
        for e in range(2):
            config["name"] = name
            config["param"] = {"alpha": np.exp(logalpha[i])}
            config["X_data"] = corticalParc
            config["weighting"] = 2
            config["train_exp"] = f"sc{e+1}"
            config["subjects"] = sn
            Model = run.train_models(config, save=True)
    pass

def train_NNLS(corticalParc, logalpha, sn=const.return_subjs):
    config = run.get_default_train_config()
    num_models = len(logalpha)
    for i in range(num_models):
        name = f"NN_{corticalParc}_A{logalpha[i]:.0f}"
        for e in range(2):
            config["name"] = name
            config["model"] = "NNLS"
            config["param"] = {"alpha": np.exp(logalpha[i])}
            config["X_data"] = corticalParc
            config["Y_data"] = 'cerebellum_suit'
            config["weighting"] = 2
            config["train_exp"] = f"sc{e + 1}"
            config["subjects"] = sn
            Model = run.train_models(config, save=True)
    pass

def make_group_data(exp = "sc1", roi="cerebellum_suit"):
    Xdata = Dataset(experiment=exp, glm="glm7", roi=roi, subj_id=const.return_subjs)
    # const.return_subjs
    Xdata.load_mat()
    Xdata.average_subj()
    Xdata.save(dataname="all")

def sum_model_eval():
    ax3 = plt.subplot(1,1,1)
    df = vis.eval_summary(exps=['sc2'])
    vis.plot_eval_predictions(dataframe=df, exps=['sc2'], methods=['WTA', 'ridge', 'lasso'], hue='eval_model', ax=ax3)
    ax3.set_xticks([80, 304, 670, 1190, 1848])

def plot_Fig2c():
    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax3=plt.subplot(1,1,1)
    df = vis.eval_summary(eval_name=['weighted_all'],exps=['sc2'],atlas=['tessels'],method=['WTA', 'ridge', 'lasso'])
    vis.plot_eval_predictions(dataframe=df, exps=['sc2'], hue='method', ax=ax3)
    ax3.set_xticks([80, 304, 670, 1190, 1848])

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


def eval_best_models(model_type=["ridge", "lasso", "WTA"],save_maps=False,eval_name='weighted_all',split='all'):
    """ Run connectivity routine (train and evaluate)

    Args:
        cortex (str): 'tesselsWB162', 'tesselsWB642' etc.
        model_type (str): 'WTA' or 'ridge' or 'NNLS'
        train_or_test (str): 'train' or 'eval'
    """
    # get best model (for each method and parcellation)
    df = vis.get_summary('train',exps='sc1')
    models, cortex_names = vis.get_best_models(df)
    # sel=['tessels' in c for c in cortex_names]
    # models = list(itertools.compress(models,sel))
    # cortex_names = list(itertools.compress(cortex_names,sel))

    for (model_name, cortex) in zip(models, cortex_names):

        dirs = const.Dirs(exp_name='sc2')

        # get default eval parameters
        config = run.get_default_eval_config()

        print(f"evaluating {model_name}")
        config["name"] = model_name
        config["X_data"] = cortex
        config["Y_data"] = 'cerebellum_suit'
        config["weighting"] = True
        config["averaging"] = "sess"
        config["train_exp"] = 'sc1'
        config["eval_exp"] = 'sc2'
        config["subjects"] = ['s02','s03','s04'] # const.return_subjs
        config["splitby"] = split
        config['incl_inst']=True
        config['save_maps']=True
        # eval model(s)
        df, voxels = run.eval_models(config)

        # save voxel data to gifti(only for cerebellum_suit)
        if save_maps:
            fpath = os.path.join(dirs.conn_eval_dir, model_name)
            cio.make_dirs(fpath)
            # Save the whole voxel structure for later usage 
            dd.io.save(os.path.join(fpath,'voxels.h5'), voxels)
            for k, v in voxels.items():
                save_maps_cerebellum(data=np.stack(v, axis=0),
                                fpath=os.path.join(fpath, f'group_{k}'))

        # eval summary
        if eval_name:
            eval_fpath = os.path.join(dirs.conn_eval_dir, f'eval_summary_{eval_name}.csv')
        else:
            eval_fpath = os.path.join(dirs.conn_eval_dir, f'eval_summary.csv')

        # concat data to eval summary (if file already exists)
        if os.path.isfile(eval_fpath):
            df = pd.concat([df, pd.read_csv(eval_fpath)])
        df.to_csv(eval_fpath, index=False)

if __name__ == "__main__":
    # D = train_NNLS('tessels0162', [-2,0,2],sn=['all'])
    # D = train_ridge('tessels0162',[-2,0,2,4,6,8],sn=['all'])
    # D = fit_group_model()
    # d = const.Dirs()
    # T = eval_models(['ridge','ridge','ridge','ridge','ridge','ridge','NN','NN','NN'],'tessels0162',[-2,0,2,4,6,8,-2,0,2],sn=['all'])
    # T.to_csv(d.conn_eval_dir / "group_model.dat")
    eval_best_models(save_maps=True)
    # df = vis.get_summary('train',exps=['sc1'],atlas=['tessels'])
    # pass
    # df = vis.get_summary('eval',summary_name="weighted_all",exps=['sc2'],atlas=['tessels'])
    # pass
    # plot_Fig2c()
    # fig.fig2()
    pass