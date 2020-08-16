# import libraries and packages
import os
import numpy as np
from time import gmtime, strftime  

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
from connectivity.models.model_functions import MODEL_MAP

"""
Created on Tue Jun 23 20:08:47 2020
Model fitting routine for connectivity models

@authors: Ladan Shahshahani and Maedbh King
"""

class TrainModel(DataManager):
    """ Model fitting Class 
        Model specifications are set in Class __init__
    """

    def __init__(self, model_name = "l2_regress"):
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
        self.model_name = model_name
        self.params = None
        self.glm = 7
        self.experiment = ['sc1']
        self.model_inputs = {'X': {'roi': 'tesselsWB162', 'file_dir': 'encoding', 'structure': 'cortex'},
                             'Y': {'roi': 'grey_nan', 'file_dir': 'encoding', 'structure': 'cerebellum'},
                            }
        self.train_mode = 'crossed'
        self.incl_inst = True
        # self.scale = True
        self.overwrite = True
        self.avg = 'run' # 'run' or 'sess'
        self.subtract_sess_mean = True
        self.subtract_exp_mean = False # not yet implemented
        self.constants = Defaults(study_name = self.experiment[0], glm = self.glm)

    def model_train(self, **kwargs):
        """ Model fitting routine on individual subject data, saved to disk
            Kwargs: 
                lambdas (list): list of lambda values. Used for example when model_name = 'l2_regress'
            Returns: 
                saves model params to JSON file format
                model weights, predictions are saved to HDF5 file format
        """

        # get model data: `X` and `Y` based on `model_inputs`
        model_data = self._get_model_data()
        
        # get indices for training data based on `train_mode`
        Xtrain_idx, Ytrain_idx = self._get_training_mode(model_data = model_data)
        
        # fit model for each `subj`
        data_all = {}
        for subj in self.constants.return_subjs:

            print(f'fitting model for s{subj:02} ...')

            # Get training data for `X` and `Y`
            xx = model_data['X']['betas'][f's{subj:02}'][Xtrain_idx] 
            yy = model_data['Y']['betas'][f's{subj:02}'][Ytrain_idx] 

            # define model
            ModelClass = MODEL_MAP[self.model_name]
            model = ModelClass(X = xx, Y = yy)

            # fit and predict model
            # model params are the same for each `sub`
            model_params, data_all[f's{subj:02}'] = model.run(**kwargs)

        # update model params
        model_params.update(self._update_model_params())

        # save model parames to JSON
        self._save_model_output(json_file = model_params, hdf5_file = data_all)
    
    def _get_model_data(self):
        """ returns model data based on `model_inputs` set in __init__
        """
        # get model data: `X` and `Y` based on `model_inputs`
        model_data = {}
        for model_input in self.model_inputs:
            
            self.data_type = self.model_inputs[model_input]
            model_data[model_input] = self.get_model_data()

        return model_data
    
    def _get_training_mode(self, model_data):
        # get sess indices for X training data
        Xtrain_idx = model_data['X']['sess']

        # get sess indices for Y training data
        Ytrain_idx = model_data['Y']['sess']

        if self.train_mode == "crossed":
            Ytrain_idx = model_data['Y']['sess'][::-1] 
        
        return Xtrain_idx.astype(int), Ytrain_idx.astype(int)
        
    def _get_outpath(self, file_type):
        """ sets outpath for connectivity training model outputs
            Args: 
                file_type (str): 'json' or 'h5' 
            Returns: 
                full path to connectivity output for model training
        """
        # define model name
        # fname     = 'mb4_%s_%s'% (rois['cortex'], model)
        X_roi = self.model_inputs['X']['roi']
        Y_roi = self.model_inputs['Y']['roi']
        timestamp = f'{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}'
        fname = f'{X_roi}_{Y_roi}_{self.model_name}_{timestamp}{file_type}'
        fpath = os.path.join(self.constants.CONN_TRAIN_DIR, fname)
        
        return fpath

    def _update_model_params(self):
       return {
            'model_name': self.model_name,
            'subjects': self.constants.return_subjs,
            'glm': self.glm,
            'train_on': self.experiment[0],
            'model_inputs': self.model_inputs,
            'train_mode': self.train_mode,
            'incl_inst': self.incl_inst,
            'average': self.avg,
            'subtract_sess_mean': self.subtract_sess_mean,
            'subtract_exp_mean': self.subtract_exp_mean
            }
    
    def _save_model_output(self, json_file, hdf5_file):
        out_path = self._get_outpath(file_type = '.json')
        io.save_dict_as_JSON(fpath = out_path, data_dict = json_file)

        # save model data to HDF5
        out_path = self._get_outpath(file_type = '.h5')
        io.save_dict_as_hdf5(fpath = out_path, data_dict = hdf5_file)
        
# run the following
model = TrainModel()
model.model_train()



        

