# import libraries and packages
import os 
import pandas as pd
import numpy as np
# import numpy.matlib
import scipy as sp
# import data_integration as di
# import essentials as es

from connectivity.constants import Defaults
from connectivity.data import utils
from connectivity.indicatormatrix import indicatorMatrix

class PrepData: 

    def __init__(self):
        self.experiments = ['sc1']
        self.sessions = [1, 2]
        self.glm = 7
        self.roi = 'grey_nan'
        self.which = 'cond'
        self.avg = 1

    def _return_Y(self):
        # get Y data for `roi`
        print('.. Y_info')
        fpath = os.path.join(self.constants.ENCODE_DIR, f's{self.subj:02}', f'Y_info_glm{self.glm}_{self.roi}.mat')
        Y = utils.read_mat_as_hdf5(fpath)['Y']['data'][:]

        return Y

    def _return_SPM_info(self):
        print('.. SPM_info')
        fpath = os.path.join(self.constants.GLM_DIR, f's{self.subj:02}', 'SPM_info.mat')
        info = utils.read_mat_as_hdf5(fpath)

        return info

    def _get_betas(self, X, Y):
        # is this correct?
        # betas = np.matmul(Y, X).T
        betas = np.matmul(np.linalg.pinv(X), Y.T)

        return betas
    
    def get_data(self):
        # loop over experiments `sc1` and `sc2`
        B_all = {} # initialise nested dict
        for exp in self.experiments:

            # get directories for `exp`
            self.constants = Defaults(study_name = exp, glm = self.glm)

            # loop over `return_subjects`
            B_subjs = {}
            for self.subj in self.constants.return_subjs:

                # return Y 
                Y = self._return_Y()

                # return SPM info
                info = self._return_SPM_info()

                # convert info dict to dataframe
                info_dataframe = utils.convert_to_dataframe(info)

                # loop over `sessions`
                B_sess = {}
                for self.sess in self.sessions:
                    B_sess[f'sess{self.sess}'] = {}

                    index = info_dataframe['cond']*(info_dataframe['sess']==self.sess)
                    X = indicatorMatrix('identity', index.values)

                    betas = self._get_betas(X, Y)

                    B_sess[f'sess{self.sess}']['betas'] = betas 

                B_subjs[f's{self.subj:02}'] = B_sess
            
            B_all[exp] = B_subjs
                
        return B_all














