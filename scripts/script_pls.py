from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run
import numpy as np
import pandas as pd
import connectivity.constants as const

def run_pls(resolution, ncomps, sn = const.return_subjs): 
    config = run.get_default_train_config()
    num_models = len(ncomps)
    for i in range(num_models):
        name = f'L2_WB{resolution}_N{ncomps[i]:.0f}'
        for e in range(2):
            config['name'] = name
            config['param'] = {'alpha':np.exp(ncomps[i])}
            config['X_data']= f'tesselsWB{resolution}'
            config['weighting'] = 2
            config['train_exp'] = e+1
            config['subjects'] = sn
            Model = run.train_models(config, save=True)
    pass

def eval_pls(resolution, ncomps, sn = const.return_subjs): 
    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(ncomps)
    D=pd.DataFrame()
    for i in range(num_models):
        name = f'L2_WB{resolution}_N{ncomps[i]:.0f}'
        for e in range(2):
            config['name'] = name
            config['logalpha'] = len(ncomps)[i] # For recording in 
            config['X_data']= f'tesselsWB{resolution}'
            config['weighting'] = 2
            config['train_exp'] = e+1
            config['eval_exp'] = 2-e
            config['subjects'] = sn
            T = run.eval_models(config)
            D = pd.concat([D,T],ignore_index = True)

    D.to_csv(d.conn_eval_dir / f'Ridge_WB{resolution}.dat')
    return D

if __name__ == '__main__':
    D = eval_pls(162,[2, 3, 4, 5, 6, 7, 8, 9, 10])
    # D = run_ridge(162,[0,2,4,6,8,10],sn=[2])