"""
Created on Tue Jun 23 20:08:47 2020
Contains functions needed for model fitting

@authors: Ladan Shahshahani and Maedbh King
"""
# import packages
import os
#import pandas as pd
import numpy as np
#import scipy as sp
#import data_integration as di
#import essentials as es
# import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from connectivity import io
from connectivity.data.prep_data import DataManager
from connectivity.constants import Defaults

class TrainModel(DataManager):
    """ Model fitting Class 
        Model specifications are set in Class __init__
    """

    def __init__(self, model = "plsregress"):
        """ Initialises inputs for TrainModel Class
            Args: 
                model (str): model name default is "plsregress"
            
            __init__:
            params (np array): model parameters # TBD
            glm (int): glm number: default is 7. options are 7 and 8
            experiment (list of str): study to be used for training. options are 'sc1' or 'sc2'
            model_inputs (nested dict): primary keys are `X` and `Y` 
            train_mode (str): training mode: 'crossed' or 'uncrossed'. If 'crossed': sessions are flipped between `X` and `Y`
            incl_instr (bool): include instructions in model fitting. default is True
            avg (str): average data across runs within a session or across session. options are 'run' and 'session'
            substract_sess_mean (bool): subract session mean. default is True
            subtract_exp_mean (bool): subtract experiment mean. default is False
        """
        super().__init__()
        self.model = model
        self.params = None
        self.glm = 7
        self.experiment = ['sc1']
        self.model_inputs = {'X': {'roi': 'tesselsWB162', 'file_dir': 'encoding'},
                             'Y': {'roi': 'grey_nan', 'file_dir': 'encoding'},
                            }
        self.train_mode = 'crossed'
        self.inclInstr = True
        # self.scale = True
        self.overwrite = True
        self.avg = 'run' # 'run' or 'sess'
        self.subtract_sess_mean = True
        self.subtract_exp_mean = False # not yet implemented

    def _get_model_data(self):
        """ returns model data based on `model_inputs` set in __init__
        """
        # get model data: `X` and `Y` based on `model_inputs`
        model_data = {}
        for input in self.model_inputs:
            
            self.data_type = self.model_inputs[input]
            model_data[input] = self.get_model_data()

        return model_data
    
    def model_fit(self):
        """ Model fitting routine on individual subject data, saved to disk
        """

        # get model data: `X` and `Y` based on `model_inputs`
        model_data = self._get_model_data()

        # get defaults
        self.constants = Defaults(study_name = self.experiment[0], glm = self.glm)
        
        # fit model for each `subj`
        for subj in self.constants.return_subjs:

             print(f'.... fitting model for s{subj:02}')

            # TO BE CONTINUED

        

