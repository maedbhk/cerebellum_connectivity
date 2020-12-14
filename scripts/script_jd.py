from connectivity.data import Dataset
import connectivity.model as model
import connectivity.run as run
import numpy as np
import pandas as pd
from connectivity.constants import Defaults, Dirs

def run_ridge(): 
    config = run.get_default_train_config()
    nameX = ['L2_WB162_A8','L2_WB162_A10']
    paramX = [{'alpha': np.exp(8)},{'alpha': np.exp(10)}]
    for i in range(len(nameX)):
        for e in range(2):
            config['name'] = nameX[i]
            config['param'] = paramX[i]
            config['weighting'] = 2
            config['train_exp'] = e+1
            config['subjects'] = [3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
            Model = run.train_models(config, save=True)
    pass

def eval_ridge(): 
    d = Dirs()
    config = run.get_default_eval_config()
    nameX = ['L2_WB162_Am2','L2_WB162_A0','L2_WB162_A2','L2_WB162_A4','L2_WB162_A6','L2_WB162_A8','L2_WB162_A10']
    logalpha = [-2.0,0.0,2.0,4.0,6.0,8.0,10.0]
    D=pd.DataFrame()
    for i in range(len(nameX)):
        for e in range(2):
            config['name'] = nameX[i]
            config['logalpha'] = logalpha[i] # For recording in 
            config['weighting'] = 2
            config['train_exp'] = e+1
            config['eval_exp'] = 2-e
            config['subjects'] = [3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
            T = run.eval_models(config)
            D = pd.concat([D,T],ignore_index = True)

    D.to_csv(d.CONN_EVAL_DIR / 'Ridge_WB162.dat')
    return D

if __name__ == '__main__':
    D = eval_ridge()