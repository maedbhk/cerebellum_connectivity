import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.data as data
import connectivity.run_mk as run
import connectivity.visualize as vis
import connectivity.figures as fig 
import itertools 

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

def eval_model(
    model_name,
    cortex,
    train_exp="sc1",
    eval_exp="sc2",
    cerebellum="cerebellum_suit",
    eval_name=None
    ):
    """Evaluate model(s)

    Args:
        model_name (str): name of trained model
        train_exp (str): 'sc1' or 'sc2'
        eval_exp (str): 'sc1' or 'sc2'
        cortex (str): cortical ROI
        cerebellum (str): cerebellar ROI
        log_locally (bool): log results locally
        eval_name (str or None): suffix to csv file for setting 
    Returns:
        Appends eval data for each model and subject into `eval_summary_<eval_name>.csv`
        Returns pandas dataframe of eval_summary
    """
    dirs = const.Dirs(exp_name=eval_exp)

    # get default eval parameters
    config = run.get_default_eval_config()

    print(f"evaluating {model_name}")
    config["name"] = model_name
    config["X_data"] = cortex
    config["Y_data"] = cerebellum
    config["weighting"] = True
    config["averaging"] = "sess"
    config["train_exp"] = train_exp
    config["eval_exp"] = eval_exp
    config["subjects"] = const.return_subjs
    config["save_maps"] = False
    config["splitby"] = "all"
    config["exclude_instruct"] = False

    # eval model(s)
    df, voxels = run.eval_models(config)

    # save voxel data to gifti(only for cerebellum_suit)
    if config["save_maps"] and config["Y_data"] == "cerebellum_suit":
        fpath = os.path.join(dirs.conn_eval_dir, model_name)
        cio.make_dirs(fpath)
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

def eval_best_models(model_type=["ridge", "lasso", "WTA"]):
    """ Run connectivity routine (train and evaluate)

    Args: 
        cortex (str): 'tesselsWB162', 'tesselsWB642' etc.
        model_type (str): 'WTA' or 'ridge' or 'NNLS'
        train_or_test (str): 'train' or 'eval'
    """
            # get best model (for each method and parcellation)
    models, cortex_names = vis.get_best_models(train_exp=f"sc1")
    sel=['tessels' in c for c in cortex_names] 
    models = list(itertools.compress(models,sel))
    cortex_names = list(itertools.compress(cortex_names,sel))

    for (model_name, cortex) in zip(models, cortex_names):
        eval_model(model_name,cortex=cortex,eval_name='weighted_all')


if __name__ == "__main__":
    # D = train_NNLS('tessels0162', [-2,0,2],sn=['all'])
    # D = train_ridge('tessels0162',[-2,0,2,4,6,8],sn=['all'])
    # D = fit_group_model()
    # d = const.Dirs()
    # T = eval_models(['ridge','ridge','ridge','ridge','ridge','ridge','NN','NN','NN'],'tessels0162',[-2,0,2,4,6,8,-2,0,2],sn=['all'])
    # T.to_csv(d.conn_eval_dir / "group_model.dat")
    eval_best_models()
    # plot_Fig2c()
    pass