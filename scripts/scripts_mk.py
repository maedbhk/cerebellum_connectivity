import numpy as np
import pandas as pd

import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run


def train_ridge(log_alpha, resolution='tesselsWB162', subj_id=const.return_subjs):
    config = run.get_default_train_config()
    cv_rmse = []
    for alpha in log_alpha:
        name = f"{resolution}_{alpha:.0f}"
        config["name"] = name
        config["param"] = {"alpha": np.exp(alpha)}
        config["X_data"] = resolution
        config["weighting"] = True
        config["train_exp"] = 'sc1'
        config["subjects"] = subj_id
        models = run.train_models(config, save=True)

        # get average cv_rmse across models
        cv_rmse.append(get_best_ridge(models))
    
    # best alpha
    best_idx = np.argmin(cv_rmse)

    return log_alpha[best_idx]


def get_best_ridge(models):
    cv_rmse = []
    # loop over subject models
    for model in models:
        cv_rmse.append(model.cv_rmse)

    return np.mean(cv_rmse)


def eval_ridge(log_alpha, resolution='tesselsWB162', subj_id=const.return_subjs):
    dirs = const.Dirs()
    config = run.get_default_eval_config()
    df_all = pd.DataFrame()
    for alpha in log_alpha:
        name = f"{resolution}_{alpha:.0f}"
        config["name"] = name
        config["log_alpha"] = alpha  # For recording in
        config["X_data"] = resolution
        config["weighting"] = True
        config["train_exp"] = 'sc1'
        config["eval_exp"] = 'sc2'
        config["subjects"] = subj_id
        df, eval_voxels = run.eval_models(config)
        df_all = pd.concat([df_all, df], ignore_index=True)

    df_all.to_csv(dirs.conn_eval_dir / f"Ridge_{resolution}.dat")
    return df_all

if __name__ == "__main__":
    best_alpha = train_ridge(log_alpha=[0,2,4,6,8,10])

    eval_ridge(log_alpha=[best_alpha])