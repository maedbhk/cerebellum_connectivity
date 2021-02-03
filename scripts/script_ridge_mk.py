import numpy as np
import pandas as pd
from random import seed, sample
import click

import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run

def get_train_test_subjects(subj_ids, test_size=.3):
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
    num_in_test = np.floor(test_size * len(subj_ids))

    # select test set
    test_subjs = list(sample(subj_ids, num_in_test))
    train_subjs = list([x for x in subj_ids if x not in test_subjs])
    
    return train_subjs, test_subjs


def train_ridge(log_alpha, resolution='tesselsWB162', subj_id=const.return_subjs, train_exp='sc1'):
    """Train ridge model(s) using different alpha values.

    Optimal alpha value is returned based on RMSE (either train or cv).

    Args: 
        log_alpha (list): list of alpha values.
        resolution (str): resolution of cortical parcellation to use as model features. 
        subj_id (list): list of subject ids (e.g., ['s01', 's02'])
        train_exp (str): 'sc1' or 'sc2'
    Returns: 
        returns alpha value that yields lowest RMSE (either train or cv ).
        saves summary RMSE data for each model and subject into `model_summary.csv`
    """
    config = run.get_default_train_config()

    cv_rmse_all = []; train_rmse_all = []
    data_out = defaultdict(list)
    # train and validate ridge models
    for alpha in log_alpha:
        print(f'training alpha {alpha:.0f}')
        name = f"ridge_{resolution}_alpha_{alpha:.0f}"
        config["name"] = name
        config["param"] = {"alpha": np.exp(alpha)}
        config["X_data"] = resolution
        config["weighting"] = True
        config["train_exp"] = train_exp
        config["subjects"] = subj_id
        config["validate_model"] = True
        config["cv_fold"] = 4

        # train model
        models = run.train_models(config, save=True)

        # get train rmse
        train_rmse, _ = _append_rmse(models, error_type='train')
        train_rmse_all.append(np.nanmean(train_rmse))

        # collect rmse for each subject and each model
        data = {'train_rmse': train_rmse, 'subj_id': subj_id, 'model_name': [name]*len(subj_id)}

        # get average cv_rmse
        if config['validate_model']:
            _, cv_rmse = _append_rmse(models, error_type='cv')
            cv_rmse_all.append(np.nanmean(cv_rmse))
            data.update({'cv_rmse': cv_rmse})

        for k, v in data.items():
            data_out[k].append(v)

    # save out rmse, alpha, subj
    dirs = const.Dirs(exp_name=train_exp)
    outpath = os.path.join(dirs.conn_train_dir, f'{train_exp}_model_summary.csv')
    
    # concat data to model_summary (if file already exists)
    if os.path.isfile(outpath):
        df = pd.read_csv(outpath)
        df = pd.concat([pd.DataFrame.from_dict(data_out), df])

    # save out model summary
    df.to_csv(outpath)
    
    # best model: lowest rmse
    best_model = _get_best_ridge(cv_rmse=cv_rmse_all, train_rmse=train_rmse_all)

    return log_alpha[best_model]


def eval_ridge(log_alpha, resolution='tesselsWB162', subj_id=const.return_subjs, train_exp='sc1', eval_exp='sc2'):
    dirs = const.Dirs()
    config = run.get_default_eval_config()
    df_all = pd.DataFrame()
    for alpha in log_alpha:
        name = f"{resolution}_{alpha:.0f}"
        config["name"] = name
        config["log_alpha"] = alpha  # For recording in
        config["X_data"] = resolution
        config["weighting"] = True
        config["train_exp"] = train_exp
        config["eval_exp"] = eval_exp
        config["subjects"] = subj_id
        
        # eval model(s)
        df, eval_voxels = run.eval_models(config)
        df_all = pd.concat([df_all, df], ignore_index=True)

    df_all.to_csv(dirs.conn_eval_dir / f"ridge_{resolution}.dat")
    return df_all


def _append_rmse(models, error_type):
    """Append rmse from model(s) to list for either train or cv.

    Args: 
        models (object class): model estimator with train_rmse and cv_rmse.
        error_type (str): either `train` or `cv`.
    Returns: 
        list of rmse for train and cv. 
    """
    cv_rmse = []; train_rmse = []
    # loop over subject models
    for model in models:
        if error_type=='train':
            train_rmse.append(model.train_rmse)
        elif error_type=='cv':
            cv_rmse.append(model.cv_rmse)

    return train_rmse, cv_rmse


def _get_best_ridge(cv_rmse, train_rmse):
    """Get idx for best ridge based on either train_rmse or cv_rmse.

    If cv_rmse is populated, this is used to determine best ridge. 
    Otherwise, train_rmse is used.

    Args: 
        cv_rmse (list): List of rmse values generated from KFold CV
        train_rmse (list): List of rmse values generated from training.
    Returns: 
        Scalar, idx for best ridge (i.e., lowest rmse)
    """
    best_idx = np.argmin(train_rmse)
    
    if cv_rmse:
        best_idx = np.argmin(cv_rmse)
    
    return best_idx


@click.command()
@click.option("--model", type=click.Choice(['train', 'eval', 'generalize']))
@click.option("--resolution")

def run(model='train', resolution='tesselsWB162'):
    click.echo(f"model={model}; resolution={resolution}")

    # split subjects into train and hold out
    train_subjs, hold_out_subjs = get_train_test_subjects(const.return_subjs)

    # loop over both experiments
    for exp in range(2):
        if model=="train":
            print("training ridge model...")
            train_ridge(log_alpha=[0,2,4,6,8,10], resolution=resolution, subj_id=['s03', 's04'], train_exp=f'sc{exp+1}')
        elif model=="eval":
            print("evaluating ridge model...")
            eval_ridge(log_alpha=[0,2,4,6,8,10], resolution=resolution, subj_id=['s03', 's04'], train_exp=f'sc{2-exp}', eval_exp=f'sc{exp+1}')
        elif model=="generalize":
            pass

if __name__ == '__main__':
    run()