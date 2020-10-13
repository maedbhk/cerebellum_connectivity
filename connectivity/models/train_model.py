# import libraries and packages
import os
import copy
import numpy as np
from time import gmtime, strftime  

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from connectivity import io
from connectivity.data.prep_data import DataManager
from connectivity.data.prep_timeseries_data import DataManager as DataManagerTS
from connectivity.constants import Defaults, Dirs
from connectivity.models.model_functions import MODEL_MAP

"""
Created on Tue Jun 23 20:08:47 2020
Model fitting routine for connectivity models

@authors: Ladan Shahshahani, Maedbh King, and Amanda LeBel
"""

class TrainModel(DataManagerTS): #not sure if the self.config would work here but a conditionel would

    def __init__(self, config, **kwargs):
        """ Model training Class, inherits methods from DataManager Class
            Model inherits method from DataManager class in prep_timeseries_data if specified'
            
            Args: 
                config (dict): dictionary loaded from `config.json` containing 
                training parameters for running connectivity models

            Kwargs:
                model_name (str): model name default is "l2_regress"
                train_sessions (list): default is [1, 2]. options are [1, 2]
                train_glm (int):  default is 7. options are 7 and 8
                train_stim (str): default is 'cond'. options are 'cond' and 'task' (depends on glm)
                train_avg (str): average over 'run' or 'sess'. default is 'run'
                train_incl_inst (bool): default is True. 
                train_subtract_sess_mean (bool): default is True.
                train_subtract_exp_mean (bool): default is True.
                train_subjects (list of int): list of subjects. see constants.py for subject list. 
                train_on (str): study to be used for training. default is 'sc1'. options are 'sc1' or 'sc2'.
                train_X_roi: 'tesselsWB362'
                train_X_file_dir: 'encoding''
                train_X_structure: 'cortex'
                train_Y_roi: 'grey_nan'
                train_Y_file_dir: 'encoding'
                train_Y_structure: 'cerebellum'
                train_mode (str): training mode: 'crossed' or 'uncrossed'. If 'crossed': sessions are flipped between `X` and `Y`. default is 'crossed'
                train_scale (bool): normalize `X` and `Y` data. default is True.
                lambdas (list of int): list of lambdas if `model_name` = 'l2_regress'
                n_pcs (list of int): list of pcs if `model_name` = 'pls_regress' # not yet implemented
        """
        self.config = copy.deepcopy(config)
        self.config.update(**kwargs)

    def model_train(self):
        """ model fitting routine on individual subject data
            model params saved to JSON and model weights, predictions are saved to HDF5
        """
        # set directories
        self.dirs = Dirs(study_name=self.config['train_on'], glm=self.config['train_glm'])

        # get model data: `X` and `Y` based on `train_inputs`
        model_data = self._get_model_data()
        
        # get indices for training data based on `train_mode`
        Xtrain_idx, Ytrain_idx = self._get_training_mode(model_data=model_data)
        
        # fit model for each `subj`
        data_all = {}
        for subj in self.config['train_subjects']:

            print(f'fitting model for s{subj:02} ...')

            # Get training data for `X` and `Y`
            xx = model_data['train_X']['betas'][f's{subj:02}'][Xtrain_idx] 
            yy = model_data['train_Y']['betas'][f's{subj:02}'][Ytrain_idx] 

            # define model
            ModelClass = MODEL_MAP[self.config['model_name']]
            if self.config['train_scale']:
                # scale `X` and `Y`
                model = ModelClass(X=self._scale_data(xx), Y=self._scale_data(yy), config=self.config)
            else:
                model = ModelClass(X=xx, Y=yy, config=self.config)

            # fit and predict model
            # model params are the same for each `sub`
            model_params, data_all[f's{subj:02}'] = model.run()

        # update model params (don't include parameters prefixed with eval)
        model_params.update({k:v for k,v in self.config.items() if not 'eval' in k})

        # save model parames to JSON and save training weights, predictions to HDF5
        self._save_model_output(json_file=model_params, hdf5_file=data_all)
    
    def _get_model_data(self):
        """ calls `get_conn_data` from `prep_data.py` based on `train_inputs` and `train_on`
            Returns: 
                model_data (dict): keys are `train_inputs` keys ('X' and 'Y')
        """
        # get model data: `X` and `Y`
        model_data = {}
        if self.config['train_stim'] == 'timeseries': #timeseries automatically returns all data; only needs to be called once
            self.data_type = {}
            self.data_type['roi'] = self.config[f'train_X_roi']
            self.data_type['file_dir'] = self.config[f'train_X_file_dir']
            self.experiment = [self.config['train_on']]
            self.glm = self.config['train_glm']
            self.stim = self.config['train_stim']
            self.subjects = self.config['train_subjects']
            self.sessions = self.config['train_sessions']
            self.number_of_delays = self.config['train_number_of_delays']
            self.detrend = self.config['train_detrend']
            self.structure = [self.config['train_X_structure'], self.config['train_Y_structure']]
            tempdata = self.get_conn_data()
            
                              
            model_data[f'train_X'] = tempdata['betas'][f'{self.config["train_X_structure"]}_undelayed'][f'{self.config["train_on"]}']
            model_data[f'train_Y'] = tempdata['betas'][f'{self.config["train_Y_structure"]}_delayed'][f'{self.config["train_on"]}']
        else:
            for model_input in ['X', 'Y']:

                # prepare variables for `prep_data`
                # this is ugly code -- clean
                self.data_type = {}
                self.data_type['roi'] = self.config[f'train_{model_input}_roi']
                self.data_type['file_dir'] = self.config[f'train_{model_input}_file_dir']
                self.experiment = [self.config['train_on']]
                self.glm = self.config['train_glm']
                self.stim = self.config['train_stim']
                self.avg = self.config['train_avg']
                self.incl_inst = self.config['train_incl_inst']
                self.subtract_sess_mean = self.config['train_subtract_sess_mean']
                self.subtract_exp_mean = self.config['train_subtract_exp_mean']
                self.subjects = self.config['train_subjects']
                self.sessions = self.config['train_sessions']

                model_data[f'train_{model_input}'] = self.get_conn_data()

        return model_data
    
    def _get_training_mode(self, model_data):
        """ gets indices for 'X' and 'Y' data based on `train_mode`
        if `train_mode` is `crossed`, then sessions are crossed between 'X' and 'Y'
            Args: 
                model_data (nested dict): contains 'X' and 'Y' data
            Returns: 
                X_indices (np array): Y_indices (np array)
        """
    
        
        if self.config['train_stim']=='timeseries':
            X_indices = str(1)
            if self.config['train_mode'] == 'crossed':
                Y_indices = str(2)
            else:
                Y_indices = str(1)
            return X_indices, Y_indices
        else:
            # get sess indices for X training data
            sessions = model_data['train_X']['sess'].astype(int)
            train_stim = self.config['train_stim']
            stims = model_data['train_X'][f'{train_stim}Num']

            X_indices = []
            for stim, sess in zip(stims, sessions):
                if sess == 1:
                    X_indices.append(stim)
                elif sess == 2:
                    X_indices.append(stim + max(stims) + 1)

            # get indices if eval_mode is `crossed`
            if self.config['train_mode'] == 'crossed':
                Y_indices = [*X_indices[-sessions.tolist().count(2):], *X_indices[:sessions.tolist().count(1)]] 
            else:
                Y_indices = X_indices

            return np.array(X_indices), np.array(Y_indices)
        
    def _get_outpath(self, file_type, **kwargs):
        """ sets outpath for connectivity training model outputs
            Args: 
                file_type (str): 'json' or 'h5' 
            Returns: 
                fpath (str): full path to connectivity output for model training
        """
        # define model name
        X_roi = self.config['train_X_roi']
        Y_roi = self.config['train_Y_roi']

        if kwargs.get('timestamp'):
            timestamp = kwargs['timestamp']
        else:
            timestamp = f'{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}'

        model_name = self.config['model_name']
        fname = f'{X_roi}_{Y_roi}_{model_name}_{timestamp}{file_type}'
        fpath = os.path.join(self.dirs.CONN_TRAIN_DIR, fname)
        
        return fpath, timestamp
    
    def _save_model_output(self, json_file, hdf5_file):
        """ saves model params to json and model data to hdf5
            Args: 
                json_file (str): json file name
                hdf5_file (str): hdf5 file name
        """
        out_path, timestamp = self._get_outpath(file_type='.json')
        io.save_dict_as_JSON(fpath=out_path, data_dict=json_file)

        # save model data to HDF5, pass in JSON timestamp (these should be the exact same)
        out_path, _ = self._get_outpath(file_type='.h5', timestamp=timestamp)
        io.save_dict_as_hdf5(fpath=out_path, data_dict=hdf5_file)
        



        

