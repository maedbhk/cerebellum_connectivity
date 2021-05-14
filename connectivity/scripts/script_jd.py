import numpy as np
import pandas as pd

import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run


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

if __name__ == "__main__":
    # D = eval_ridge('tessels0162', [0])
    D = train_ridge('tessels0162',[-2,0,2,4,6,8])
    # D = train_NNLS('tessels0162',[-2],['s02'])