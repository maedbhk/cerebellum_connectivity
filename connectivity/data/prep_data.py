# import libraries and packages
import os 
import pandas as pd
import numpy as np
import scipy as sp
import re

from connectivity.constants import Defaults
from connectivity import io
from connectivity.indicatormatrix import indicatorMatrix

"""
Created on Fri Jun 19 11:48:47 2020
Used for preparing data for connectivity modelling and evaluation

@authors: Ladan Shahshahani and Maedbh King
"""

class PrepBetas: 

    def __init__(self):
        self.experiments = ['sc1', 'sc2'] # ['sc1', 'sc2']
        self.sessions = [1, 2]
        self.glm = 7
        self.roi = 'grey_nan'
        self.stim = 'cond' # 'cond' or 'task'
        self.avg = 'run' # 'run' or 'sess'

    def _get_Y(self):
        # get Y data for `roi`
        print('.. Y')
        return self.Y_info['data'][:]

    def _get_X(self):
        # get stim and sess info from Y_info
        print('.. X')

        stim = self.Y_info[self.stim].value.flatten()  # `cond` or `task`
        sess = self.Y_info['sess'].value.flatten() 

        index = stim*(sess==self.sess)
        X = indicatorMatrix('identity', index)

        return X
    
    def _read_SPM_info(self):
        # DEPRECIATED, we are now getting this info from Y_info
        print('.. SPM_info')
        fpath = os.path.join(self.constants.GLM_DIR, f's{self.subj:02}', 'SPM_info.mat')
        info = io.read_mat_as_hdf5(fpath)

        return info

    def _read_Y_info(self):
        fpath = os.path.join(self.constants.ENCODE_DIR, f's{self.subj:02}', f'Y_info_glm{self.glm}_{self.roi}.mat')
        return io.read_mat_as_hdf5(fpath)['Y']
    
    def _add_task_conds(self):
        """ filters task_conds dataframe and returns dict
        """
        dataframe = pd.read_csv(os.path.join(self.constants.BASE_DIR, self.constants.conn_file), sep = '\t')
                
        exp_num = int((re.findall('\d+', self.exp))[0])  

        def strip(dataframe):
            for col in dataframe.columns:
                try: 
                    dataframe[col] = dataframe[col].str.strip()
                except: AttributeError
            return dataframe

        # remove trailing spaces from str cols
        dataframe = strip(dataframe)

        return dataframe.query(f'StudyNum=={exp_num}').groupby(f'{self.stim}Num').first().to_dict(orient='list')
    
    def _calculate_betas(self, X, Y):
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
            print('choose a valid glm')
    
    def _get_outpath(self):
        # save dict to disk as HDF5 file obj
        if self.avg=='run':
            out_name = f'mbeta_{self.roi}_all.h5'
        elif self.avg=='sess':
            out_name = f'beta_{self.roi}_all.h5'

        out_path = os.path.join(self.constants.ENCODE_DIR, out_name)
        
        return out_path

    def get_betas(self):
        """ calculates average betas across runs/sessions
            for exp, subj, sess for voxel/roi data
            Returns: 
                saves data dict with averaged betas as HDF5 file
        """

        # check that we're using correct stim
        self._check_glm_type()

        # loop over experiments `sc1` and `sc2` and save to disk
        for self.exp in self.experiments:
            B_exp = {}

            # get directories for `exp`
            self.constants = Defaults(study_name = self.exp, glm = self.glm)

            # add task info to nested dict
            B_exp[self.exp] = self._add_task_conds()

            # loop over `return_subjects`
            B_exp[self.exp]['betas'] = {}
            B_subjs = {}
            for self.subj in self.constants.return_subjs:

                # get Y_info
                self.Y_info = self._read_Y_info()

                # get Y
                Y = self._get_Y()

                # loop over `sessions`
                B_sess = {}
                for self.sess in self.sessions:

                    # return design matrix
                    X = self._get_X()

                    # calculate betas
                    B_sess[f'sess{self.sess}'] = self._calculate_betas(X, Y)
            
                B_subjs[f's{self.subj:02}'] = B_sess

            B_exp[self.exp]['betas'] = B_subjs

            # save dict as HDF5 file for each `exp`
            io.save_dict_as_hdf5(fpath = self._get_outpath(), data_dict = B_exp)

    def get_wcon(self):
        pass
        # TO DO

prep = PrepBetas()
B_exp = prep.get_betas()