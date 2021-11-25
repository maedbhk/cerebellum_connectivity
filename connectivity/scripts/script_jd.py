import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.data as data
import connectivity.run as run
import connectivity.visualize as vis
import connectivity.figures as fig 


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


def eval_ridge(corticalParc, logalpha, sn=const.return_subjs):
    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(logalpha)
    D = pd.DataFrame()
    for i in range(num_models):
        name = f"ridge_{corticalParc}_A{logalpha[i]:.0f}"
        for e in range(2):
            config["name"] = name
            config["logalpha"] = logalpha[i]  # For recording in
            config["X_data"] = corticalParc
            config["weighting"] = 2
            config["train_exp"] = f'sc{e + 1}'
            config["eval_exp"] = f'sc{2 - e}'
            config["subjects"] = sn
            T = run.eval_models(config)
            D = pd.concat([D, T], ignore_index=True)

    D.to_csv(d.conn_eval_dir / f"Ridge_{corticalParc}.dat")
    return D

def eval_models(model_name, corticalParc, logalpha, sn=const.return_subjs):
    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(logalpha)
    if type(corticalParc) is not list: 
        corticalParc = [corticalParc] * num_models
    if type(model_name) is not list: 
        model_name = [model_name] * num_models

    D = pd.DataFrame()
    for model,cort,loga in zip(model_name,corticalParc,logalpha):
        name = f"{model}_{cort}_A{loga:.0f}"
        for e in range(2):
            config["name"] = name
            config["model"] = model
            config["logalpha"] = loga  # For recording in
            config["X_data"] = cort
            config["weighting"] = 2
            config["train_exp"] = f'sc{e + 1}'
            config["eval_exp"] = f'sc{2 - e}'
            config["subjects"] = sn
            T = run.eval_models(config)
            D = pd.concat([D, T], ignore_index=True)

    return D


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


if __name__ == "__main__":
    # D = train_NNLS('tessels0162', [-2,0,2],sn=['all'])
    # D = train_ridge('tessels0162',[-2,0,2,4,6,8],sn=['all'])
    # D = fit_group_model()
    # d = const.Dirs()
    # T = eval_models(['ridge','ridge','ridge','ridge','ridge','ridge','NN','NN','NN'],'tessels0162',[-2,0,2,4,6,8,-2,0,2],sn=['all'])
    # T.to_csv(d.conn_eval_dir / "group_model.dat")
    x_pos = -0.1
    y_pos = 1.1
    labelsize = 30

    ax3=plt.subplot(1,1,1)
    df = vis.eval_summary(exps=['sc2'])
    vis.plot_eval_predictions(dataframe=df, exps=['sc2'], methods=['WTA', 'ridge', 'lasso'], hue='model', ax=ax3)
    ax3.set_xticks([80, 304, 670, 1190, 1848])
    pass