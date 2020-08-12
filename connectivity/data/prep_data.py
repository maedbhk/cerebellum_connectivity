# import libraries and packages
import os 
import pandas as pd
import numpy as np
import scipy as sp
import re
import deepdish as dd
import copy

from connectivity.constants import Defaults
from connectivity import io
from connectivity.indicatormatrix import indicatorMatrix

"""
Created on Fri Jun 19 11:48:47 2020
Prepares data for connectivity modelling and evaluation

@authors: Ladan Shahshahani and Maedbh King
"""

class DataManager: 

    def __init__(self):
        self.experiment = ['sc1', 'sc2'] # ['sc1', 'sc2']
        self.sessions = [1, 2] # [1, 2]
        self.glm = 7 # 7 or 8
        self.data_type = {'roi': 'grey_nan', 'file_dir': 'encoding'} # cerebellum_grey (when 'beta_roi'), 'grey_nan' (when 'encoding') tesselsWB162, tesselsWB362, tesselsWB642
        self.stim = 'cond' # 'cond' or 'task'
        self.avg = 'run' # 'run' or 'sess'
        self.subtract_sess_mean = True # True of False
        self.subtract_exp_mean = False # not yet implemented

    def get_model_data(self):
        """ prepares data for modelling based on specifications set in __init__
            calls `get_betas` if file has not been saved to disk
            Returns: 
                B_all (dict): keys are info (e.g. StudyNum) and betas, concatenated across `sessions` and `experiment`
        """

        # check that we're setting correct parameters
        self._check_init()

        # return `exp` data
        B_dict = self._concat_exps()

        # return concatenated info based on `B_concat`
        info_dict = self._concat_info(exps_dict = copy.deepcopy(B_dict)) # need to do a deepcopy here

        B_subjs = {}
        B_all = {}
        for self.subj in self.constants.return_subjs:

            betas = []
            sessions = []
            for self.exp in self.experiment:

                # get `exp`
                B_exp = B_dict[self.exp]

                # get `subj` `betas` dict 
                B_subj = B_exp['betas'][f's{self.subj:02}']
                
                for self.sess in self.sessions:

                    if self.subtract_sess_mean:
                        # subtract `sess` mean
                        sess_mean = np.nanmean( B_subj[f'sess{self.sess}'], axis = 0)
                        sess_betas =  B_subj[f'sess{self.sess}'] - sess_mean
                    else:
                        sess_betas =  B_subj[f'sess{self.sess}']

                    # append betas across `subj` and `sess`
                    betas.append(sess_betas)
                    sessions.append(np.ones(len(B_exp['StudyNum'])) * self.sess)

            # vertically stack `betas` across `sess` and `exp`
            B_subjs[f's{self.subj:02}'] = np.vstack(betas)

        # add `info`, `betas`, and `sess` to `B_all`
        B_all = info_dict
        B_all['betas'] = B_subjs
        B_all['sess'] = np.concatenate(sessions)

        return B_all
    
    def prep_betas(self):
        """ calculates average betas across runs/sessions
            for exp, subj, sess for voxel/roi data
            Returns: 
                saves data dict with averaged betas as HDF5 file
        """
        # check that we're setting correct parameters
        self._check_init()

        # loop over experiment `sc1` and `sc2` and save to disk
        B_exp = {}
        for self.exp in self.experiment:
            # get directories for `exp`
            self.constants = Defaults(study_name = self.exp, glm = self.glm)

            # add task info to nested dict
            B_exp[self.exp] = self._add_task_conds()

            # loop over `return_subjects`
            B_exp[self.exp]['betas'] = {}
            B_subjs = {}
            for self.subj in self.constants.return_subjs:
                print(f'prepping betas for s{self.subj:02}')

                # get Y_info
                self.Y_info = self._read_Y_data()

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
            io.save_dict_as_hdf5(fpath = self._get_path_to_betas(), data_dict = B_exp)
        
    def _get_Y(self):
        # get Y data for `roi`
        print(f'.. Y for {self.exp} and s{self.subj:02}')
        return self.Y_info['data'][:]

    def _get_X(self):
        print('_get_X')
        # get stim and sess info from Y_info
        print(f'.. X for {self.exp}, s{self.subj:02}, sess{self.sess}')

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

    def _read_Y_data(self):
        file_dir = self.data_type['file_dir']  
        roi = self.data_type['roi']
        if file_dir == 'encoding':
            fpath = os.path.join(self.constants.ENCODE_GLM_DIR, f's{self.subj:02}', f'Y_info_glm{self.glm}_{roi}.mat')
        elif file_dir == 'beta_roi':
            fpath = os.path.join(self.constants.BETA_REG_GLM_DIR, f's{self.subj:02}', f'Y_info_glm{self.glm}_{roi}.mat')
        return io.read_mat_as_hdf5(fpath)['Y']
    
    def _add_task_conds(self):
        """ filters task_conds dataframe and returns dict
        takes the first `stim` from each group, relevant for `instruct`
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

        return dataframe.query(f'StudyNum=={exp_num}').groupby(f'{self.stim}Num').first().to_dict(orient = 'list')
    
    def _calculate_betas(self, X, Y):
        # is this correct?
        # betas = np.linalg.pinv(X) @ Y[0:X.shape[0],:]
        betas = np.matmul(np.linalg.pinv(X), Y.T)

        return betas
    
    def _check_init(self):
        if self.glm==7:
            self.stim = 'cond'
        elif self.glm==8:
            self.stim = 'task'
        else:
            print('choose a valid glm')
        
        roi = self.data_type['roi']
        if roi == 'cerebellum_grey':
            self.data_type['file_dir'] = 'beta_roi'
        elif roi == 'grey_nan':
            self.data_type['file_dir'] = 'encoding'
    
    def _get_path_to_betas(self):
        # save dict to disk as HDF5 file obj
        roi = self.data_type['roi']
        if self.avg=='run':
            fname = f'mbeta_{roi}_all.h5'
        elif self.avg=='sess':
            fname = f'beta_{roi}_all.h5'

        if self.data_type['file_dir'] == 'encoding':
            fpath = os.path.join(self.constants.ENCODE_GLM_DIR, fname)
        elif self.data_type['file_dir'] == 'beta_roi':
            fpath = os.path.join(self.constants.BETA_REG_GLM_DIR, fname)
        
        return fpath

    def _concat_exps(self):
        B_concat = {}
        for exp in self.experiment:

            # get directories for `exp`
            self.constants = Defaults(study_name = exp, glm = self.glm)

            # load betas file for `exp`
            fpath = self._get_path_to_betas()

            # create betas for `experiment` if they don't exist
            if not os.path.isfile(fpath):
                self.prep_betas()
            
            # load betas from file
            B_concat[exp] = dd.io.load(fpath)[exp]

        return B_concat
    
    def _concat_info(self, exps_dict):  
        """ takes the info keys from `exps_dict` and concats them across 
            `sess` and `exp` and return dict
            Args: 
                exps_dict (dict): nested dictionary with `exp` as key
            Returns: 
                dict where info cols (i.e. StudyNum etc) are concat across `sess` and `exp`
        """        
        # remove `betas` from temporary dict and convert `B_exp` to dict
        dataframes_all = pd.DataFrame()
        for exp in exps_dict.keys():
            exps_dict[exp].pop('betas')
            dataframe_concat = pd.concat([pd.DataFrame.from_dict(exps_dict[exp]) ]*len(self.sessions)) 
            dataframes_all = pd.concat([dataframes_all, dataframe_concat])

        return pd.DataFrame.to_dict(dataframes_all, orient = 'list')       

# run the following to return model data
# all inputs are set in __init__
prep = DataManager()
model_data = prep.get_model_data()