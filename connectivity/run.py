import os
import sys
import glob
import numpy as np

import click

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

def get_default_train_config:
    # defaults training config: 
    config = {
        "name": 'model1',
        'model': model.L2regression,
        'param': {'alpha': 1},
        "sessions": [1, 2],
        "glm": 7,
        "avgeraging": "sess",
        "incl_inst": True,
        'X_data': 'cerebellum_grey',
        'Y_data': 'tessels_WB162',
        "subjects": [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31],
        "mode": "crossed",
    }
    return config

def train_models(modelClass, config, save = True) 
    """ 
        train_model: trains a specific model class on X and Y data from a specific experiment for all subjects 
        Parameters: 
            config (dict)
                Training configuration (see train_config)
        Returns: 
            models (list)
                List of trained models for all subject 
    """ 
    models = [] 
    for s in config['subjects']
        Ydata=Dataset(glm =config['glm'], sn = s, roi=config['Y_data'])
        Ydata.load_mat()
        Y,T = Ydata.get_data(averaging=config['averaging'])
        Xdata = Dataset(glm=config['glm'], sn= s, roi=config['X_data']) 
        models
        Xdata.load_mat()
        X,T = Xdata.get_data(averaging=config['averaging'])

        # Generate new model and put in the list 
        models.append(config['models']) 
        # Fit this model 
        models[-1].fit(X,Y)
    return models

def eval_models(models, config):
    """ This routine does model evaluation
        Args: 
            config (dict): dictionary with specific parameters 
    """
    # evaluate
    pass
