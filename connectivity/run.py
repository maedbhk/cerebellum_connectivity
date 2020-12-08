import os
import sys
import glob
import numpy as np
import json
import deepdish as dd 

from connectivity import io
import connectivity.model as model 
from connectivity.data import Dataset
from connectivity.constants import Defaults, Dirs

import warnings
warnings.filterwarnings('ignore')

np.seterr(divide='ignore', invalid='ignore')

"""
Created on Aug 31 11:14:19 2020
Main script for training and evaluating connectivity models

@authors: Maedbh King, Jörn Diedrichsen 
"""

def _delete_conn_files():
    """ delete any pre-existing connectivity output
    """
    for exp in ['sc1', 'sc2']:
        dirs = Dirs(study_name=exp, glm=7)
        filelists = [glob.glob(os.path.join(dirs.CONN_TRAIN_DIR, '*')), glob.glob(os.path.join(dirs.CONN_EVAL_DIR, '*'))]
        for filelist in filelists:
            for f in filelist:
                os.remove(f)
    print('deleting training and results data')

def get_default_train_config():
    # defaults training config: 
    config = {
        "name": 'L2_WB162_A1',
        'model': 'L2regression',
        'param': {'alpha': 1},
        "sessions": [1, 2],
        "glm": 7,
        "exp": 1,
        "averaging": "sess",
        "incl_inst": True,
        'X_data': 'cerebellum_grey',
        'Y_data': 'tesselsWB162',
        "subjects": [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31],
        "mode": "crossed",
        "save": True
    }
    return config

def get_default_eval_config():
    # defaults training config: 
    config = {
        "name": 'L2_WB162_A1',
        "sessions": [1, 2],
        "glm": 7,
        "train_exp": 1,
        "eval_exp": 2,
        "averaging": "sess",
        "incl_inst": True,
        'X_data': 'cerebellum_grey',
        'Y_data': 'tesselsWB162',
        "subjects": [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31],
        "mode": "crossed",
        "save_pred": False,
        "eval_splitby": None
    }
    return config

def train_models(config, save = False):
    """ 
        train_model: trains a specific model class on X and Y data from a specific experiment for all subjects 
        Parameters: 
            config (dict)
                Training configuration (see get_default_train_config)
            save (bool)
                Save fitted models automatically to conn-folder
        Returns: 
            models (list)
                List of trained models for all subject 
    """
    exp = config['exp']
    dirs = Dirs(study_name=f'sc{exp}', glm=config['glm'])
    models = []
    if save:
        fname = [config['name']]
        fpath = dirs.CONN_TRAIN_DIR / config['name'] / 'train_config.json'
        with open(fpath, 'w') as fp:
            json.dump(config, fp)
    for s in config['subjects']:
        print(f'Subject{s:02d}\n')
        Ydata=Dataset(glm =config['glm'], sn = s, roi=config['Y_data'])
        Ydata.load_mat()
        Y,T = Ydata.get_data(averaging=config['averaging'])
        Xdata = Dataset(glm=config['glm'], sn= s, roi=config['X_data']) 
        models
        Xdata.load_mat()
        X,T = Xdata.get_data(averaging=config['averaging'])

        # Generate new model and put in the list
        newModel = getattr(model,config['model'])()
        models.append(newModel)
        # Fit this model 
        models[-1].fit(X,Y)
        
        # Save the fitted model to disk if required 
        if save: 
            fname = _get_model_name(config['name'],config['exp'],s)
            dd.io.save(fname,models[-1], compression = None) 
    # Return list of models
    return models

def eval_models(config):
    """ 
        eval_models: trains a specific model class on X and Y data from a specific experiment for all subjects 
        Parameters: 
            config (dict)
                Eval configuration (see get_default_eval_config
        Returns: 
            T (pd.DataFrame)
                Evaluation of different models on the data 
    """

    texp = config['train_exp']
    eexp = config['eval_exp']
    tdirs = Dirs(study_name=f'sc{texp}', glm=config['glm'])
    edirs = Dirs(study_name=f'sc{eexp}', glm=config['glm'])

    for s in config['subjects']:
        print(f'Subject{s:02d}\n')

        # Get the data
        Ydata=Dataset(glm =config['glm'], sn = s, roi=config['Y_data'])
        Ydata.load_mat()
        Y,T = Ydata.get_data(averaging=config['averaging'])
        Xdata = Dataset(glm=config['glm'], sn= s, roi=config['X_data']) 
        Xdata.load_mat()
        X,T = Xdata.get_data(averaging=config['averaging'])

        # Get the model from file
        fname = _get_model_name(config['name'],config['exp'],s)
        model = dd.io.load(fname)

        # Save the fitted model to disk if required 
        Ypred = model.predict(X)
        if config['mode']=='crossed':
            Ypred=np.r_[Ypred(T.sess==2,:),Ypred(T.sess==1,:)]
        
        # Copy over all scalars or strings to the Data frame: 
        for key, value in config:
            if type(value) is not list:
                T['key']=value
        
        # Add the subject number 
        T['SN'] = s
        
        T['R'], Rvox[s,L] = eval(Y,Ypred)

    # Return list of models
    return models

def _get_model_name(train_name, exp, subject):
    """ returns path/name for connectivity training model outputs
        Args: 
            train_name (str)
                Name of trained model
            exp (int)
                Experiment number
            subject (int)
                Subject number
        Returns: 
            fpath (str)
                full path and name to connectivity output for model training
    """
    dirs = Dirs(study_name=f'sc{exp}')
    fname = f'{train_name}_s{subject:02d}.h5'
    fpath = os.path.join(dirs.CONN_TRAIN_DIR, train_name, fname)
    return fpath
    