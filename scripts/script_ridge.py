from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run
import numpy as np
import pandas as pd
import connectivity.constants as const

def run_ridge(resolution, logalpha, sn = const.return_subjs): 
    config = run.get_default_train_config()
    num_models = len(logalpha)
    for i in range(num_models):
        name = f'L2_WB{resolution}_A{logalpha[i]:.0f}'
        for e in range(2):
            config['name'] = name
            config['param'] = {'alpha':np.exp(logalpha[i])}
            config['X_data']= f'tesselsWB{resolution}'
            config['weighting'] = 2
            config['train_exp'] = e+1
            config['subjects'] = sn
            Model = run.train_models(config, save=True)
    pass

def eval_ridge(resolution, logalpha, sn = const.return_subjs): 
    d = const.Dirs()
    config = run.get_default_eval_config()
    num_models = len(logalpha)
    D=pd.DataFrame()
    for i in range(num_models):
        name = f'L2_WB{resolution}_A{logalpha[i]:.0f}'
        for e in range(2):
            config['name'] = name
            config['logalpha'] = logalpha[i] # For recording in 
            config['weighting'] = 2
            config['train_exp'] = e+1
            config['eval_exp'] = 2-e
            config['subjects'] = sn
            T = run.eval_models(config)
            D = pd.concat([D,T],ignore_index = True)

    D.to_csv(d.conn_val_dir / f'Ridge_WB{resolution}.dat')
    return D

if __name__ == '__main__':
    D = run_ridge(642,[-2,0,2,4,6,8,10])
    # D = run_ridge(162,[-2],[2])