# import libraries and packages
import os 
import pandas as pd
import numpy as np
import scipy as sp

from connectivity.constants import Defaults
from connectivity.data import utils
from connectivity.indicatormatrix import indicatorMatrix

"""
Created on Fri Jun 19 11:48:47 2020
Used for preparing data for connectivity modelling and evaluation

@authors: Maedbh King and Ladan Shahshahani
"""

class PrepData: 

    def __init__(self):
        self.experiments = ['sc1'] # ['sc1', 'sc2']
        self.sessions = [1, 2]
        self.glm = 7
        self.roi = 'grey_nan'
        self.stim = 'cond' # 'cond' or 'task'
        self.avg = 'run' # 'run' or 'sess'

    def _get_Y(self):
        # get Y data for `roi`
        print('.. Y_info')
        fpath = os.path.join(self.constants.ENCODE_DIR, f's{self.subj:02}', f'Y_info_glm{self.glm}_{self.roi}.mat')
        Y = utils.read_mat_as_hdf5(fpath)['Y']['data'][:]

        return Y

    def _get_X(self):
        index = self.dataframe[self.stim]*(self.dataframe['sess']==self.sess)
        X = indicatorMatrix('identity', index.values)

        return X
    
    def _get_SPM_info(self):
        print('.. SPM_info')
        fpath = os.path.join(self.constants.GLM_DIR, f's{self.subj:02}', 'SPM_info.mat')
        info = utils.read_mat_as_hdf5(fpath)

        return info

    def _get_betas(self, X, Y):
        # is this correct?
        # betas = np.linalg.pinv(X) @ Y[0:X.shape[0],:]
        betas = np.matmul(np.linalg.pinv(X), Y.T)

        return betas
    
    def _check_glm_type(self):
        if self.glm==7:
            self.stim = 'cond'
        elif self.glm==8:
            self.stim = 'task'
        else:
            print('choose a glm')
    
    def get_data(self):
        """ calculates average betas across runs/sessions
            for exp, subj, sess for voxel/roi data
            Returns: 
                saves data dict with averaged betas as HDF5 file
        """

        # check that we're using correct stim
        self._check_glm_type()

        # loop over experiments `sc1` and `sc2`
        B_all = {} # initialise nested dict
        for exp in self.experiments:

            # get directories for `exp`
            self.constants = Defaults(study_name = exp, glm = self.glm)

            # loop over `return_subjects`
            B_subjs = {}
            for self.subj in self.constants.return_subjs:

                # return Y 
                Y = self._get_Y()

                # return SPM info
                info = self._get_SPM_info()

                # convert info dict to dataframe
                self.dataframe = utils.convert_to_dataframe(info)

                # loop over `sessions`
                B_sess = {}
                for self.sess in self.sessions:
                    B_sess[f'sess{self.sess}'] = {}

                    # return design matrix
                    X = self._get_X()

                    # calculate betas
                    betas = self._get_betas(X, Y)

                    B_sess[f'sess{self.sess}']['betas'] = betas 

                B_subjs[f's{self.subj:02}'] = B_sess

                # add task info to nested dict
            
            B_all[exp] = B_subjs
                
        return B_all














