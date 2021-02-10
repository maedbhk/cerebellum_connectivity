import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import connectivity.constants as const 
import connectivity.io as cio

plt.rcParams["axes.grid"] = False

def train_summary(exps=['sc1', 'sc2'], summary_name='train_summary'):
    """
    """
    # look at model summary for train results
    df_concat = pd.DataFrame()
    for exp in exps:
        dirs = const.Dirs(exp_name=exp)
        fpath = os.path.join(dirs.conn_train_dir, f'{summary_name}.csv')
        df = pd.read_csv(fpath)
        # df['train_exp'] = exp
        df_concat = pd.concat([df_concat, df]) 

    # rename cols
    cols = []
    for col in df_concat.columns:
        if 'train' not in col:
            cols.append('train_' + col)
        else:
            cols.append(col)

    df_concat.columns = cols

    return df_concat


def eval_summary(exps=['sc1', 'sc2'], summary_name='eval_summary'):
    # look at model summary for eval results
    df_concat = pd.DataFrame()
    for exp in exps:
        dirs = const.Dirs(exp_name=exp)
        fpath = os.path.join(dirs.conn_eval_dir, f'{summary_name}.csv')
        df = pd.read_csv(fpath)
        # df['exp'] = exp
        df_concat = pd.concat([df_concat, df])
    
    cols = []
    for col in df_concat.columns:
        if any(s in col for s in ('eval', 'train')):
            cols.append(col)
        else:
            cols.append('eval_' + col)

    df_concat.columns = cols

    return df_concat


def plot_train_predictions(dataframe, hue=None):
    
    plt.figure(figsize=(8,8))

    # R
    sns.factorplot(x='train_alpha', y='train_R_cv', hue=hue, data=dataframe, legend=False)
    plt.title('Model Training (CV Predictions)', fontsize=20);
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('log alpha', fontsize=20)
    plt.ylabel('R', fontsize=20);
    plt.legend(fontsize=15)


def plot_eval_predictions(dataframe, best_alpha=None, hue=None):

    # make sure you're only plotting your chosen alpha
    if best_alpha:
        dataframe = dataframe.query(f'eval_alpha=={best_alpha}')
    
    # get noise ceilings
    dataframe['eval_noiseceiling_Y'] = np.sqrt(dataframe.eval_noise_Y_R)
    dataframe['eval_noiseceiling_XY'] = np.sqrt(dataframe.eval_noise_Y_R)*np.sqrt(dataframe.eval_noise_X_R)
    
    # melt data into one column for easy plotting
    cols = ['eval_noiseceiling_Y', 'eval_noiseceiling_XY', 'R_eval']
    df = pd.melt(dataframe, value_vars=cols, id_vars=set(dataframe.columns)-set(cols)).rename({'variable':'data_type','value':'data'}, axis=1)

    plt.figure(figsize=(8,8))

    sns.barplot(x='data_type', y='data', hue=hue, data=df)
    plt.title(f'Model Evaluation (log alpha={best_alpha})', fontsize=20);
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('')
    plt.ylabel('R', fontsize=20);
    plt.xticks([0,1,2], ['noise ceiling (data)', 'noise ceiling (model)', 'model predictions'], rotation ='45')
    plt.legend(fontsize=15)