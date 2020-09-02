# import libraries and packages
import os 
import pandas as pd
import numpy as np
import scipy as sp
import re
import deepdish as dd
import copy

from sklearn.preprocessing import StandardScaler

from connectivity.constants import Defaults, Dirs
from connectivity import io
from connectivity.indicatormatrix import indicatorMatrix

"""
Created on Fri Jun 19 11:48:47 2020
Prepares data for connectivity modelling and evaluation

@authors: Ladan Shahshahani and Maedbh King
"""

class DataManager: 
    """ Data manager class, preps betas for connectivity modelling
        Initialises inputs for DataManager Class:
            experiment (list): default is ['sc1', 'sc2']. options are ['sc1', 'sc2']
            sessions (list): default is [1, 2]. options are [1, 2]
            glm (int):  default is 7. options are 7 and 8
            stim (str): default is 'cond'. options are 'cond' and 'task' (depends on glm)
            data_type (dict): default is {'roi': 'grey_nan', 'file_dir': 'encoding'}
            avg (str): average over 'run' or 'sess'. default is 'run'
            incl_inst (bool): default is True. 
            subtract_sess_mean (bool): default is True.
            subtract_exp_mean (bool): default is True.
            subjects (list of int): list of subjects. see constants.py for subject list. 
    """

    def __init__(self):
        self.experiment = ['sc1', 'sc2']
        self.sessions = [1, 2]
        self.glm = 7
        self.stim = 'cond'
        self.data_type = {'roi': 'grey_nan', 'file_dir': 'encoding'}
        self.avg = 'run'
        self.incl_inst = True
        self.subtract_sess_mean = True
        self.subtract_exp_mean = True
        self.subjects = [3, 4]

    def get_conn_data(self):
        """ prepares data for modelling and evaluation
            calls `prep_betas` if file has not been saved to disk
            Returns: 
                B_all (nested dict): keys are info (i.e. task_num, study_num)
                and betas concatenated across `sessions` and `experiment`
        """

        # check that we're setting correct parameters
        self._check_init()

        # return `exp` data
        B_dict = self._concat_exps()

        # return concatenated info based on `B_concat`
        info_dict = self._concat_info(exps_dict = copy.deepcopy(B_dict)) # need to do a deepcopy here

        B_subjs = {}
        B_all = {}
        for self.subj in self.subjects:

            betas = []
            sessions = []
            for self.exp in self.experiment:

                # get `exp`
                B_exp = B_dict[self.exp]

                # get `subj` `betas` dict 
                B_subj = B_exp['betas'][f's{self.subj:02}']
                
                for self.sess in self.sessions:

                    # subtract sess mean
                    sess_betas =  B_subj[f'sess{self.sess}']
                    if self.subtract_sess_mean:
                        # subtract `sess` mean
                        sess_betas = self._subtract_mean(arr = sess_betas, axis = 0)

                    # append betas across `subj` and `sess`
                    betas.append(sess_betas)
                    sessions.append(np.ones(len(B_exp['StudyNum'])) * self.sess)
                
                # subtract exp mean
                exp_betas = np.vstack(betas)
                if self.subtract_exp_mean:
                    exp_betas = self._subtract_mean(arr = exp_betas, axis = 0)

            # vertically stack `betas` across `sess` and `exp`
            B_subjs[f's{self.subj:02}'] = exp_betas

        # add `info`, `betas`, and `sess` to `B_all`
        B_all = info_dict
        B_all['betas'] = B_subjs
        B_all['sess'] = np.concatenate(sessions)

        return B_all
    
    def prep_betas(self):
        """ calculates average betas across runs or sessions
            for exp, subj, sess, for voxel or roi data
            Returns: 
                saves data dict with averaged betas as HDF5 file
        """
        # check that we're setting correct parameters
        self._check_init()

        # loop over experiment `sc1` and `sc2` and save to disk
        B_exp = {}
        for self.exp in self.experiment:

            # get directories for `exp`
            self.dirs = Dirs(study_name = self.exp, glm = self.glm)

            # add task info to nested dict
            B_exp[self.exp] = self._add_task_conds()

            # loop over `return_subjects`
            B_exp[self.exp]['betas'] = {}
            B_subjs = {}
            for self.subj in self.subjects:
                print(f'prepping betas for s{self.subj:02} ...')

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
        
    def _scale_data(self, X):
        """ standardize features by removing the mean and scaling to unit variance
            Args:
                X (array-like, sparse matrix): shape [n_samples, n_features]
            Returns: 
                standarized (centered and scaled) X
        """
        scaler = StandardScaler() # define Class
        scaler.fit(X) # computes the mean and std to be used for later scaling.
        return scaler.transform(X) # perform standardization by centering and scaling
    
    def _subtract_mean(self, arr, axis):
        """ subtract mean from data array
            Args: 
                arr (array_like): array containing numbers whose mean is desired
                axis (int, None): Axis or axes along which the means are computed (0: col, 1: row, None: flattened array)
            Returns: 
                mean-subtracted array
        """
        arr_mean = np.nanmean(arr, axis = axis)
        return arr - arr_mean

    def _get_Y(self):
        """ returns Y data from `Y_info` file
        """
        # get Y data for `roi`
        print(f'getting Y for {self.exp} and s{self.subj:02} ...')
        return self.Y_info['data'][:]

    def _get_X(self):
        """ calculates X design matrix using stims
        and sess info from `Y_info` file
            Returns: 
                X (matrix): design matrix
        """
        # get stim and sess info from Y_info
        print(f'getting X for {self.exp}, s{self.subj:02}, sess{self.sess} ...')

        stim = self.Y_info[self.stim].value.flatten()  # `cond` or `task`
        sess = self.Y_info['sess'].value.flatten() 

        index = stim*(sess==self.sess)
        X = indicatorMatrix('identity', index)

        return X

    def _read_Y_data(self):
        """ Returns:
                Y_info (dict): Y info data dict from either `encoding` or `beta_roi` dirs
        """
        file_dir = self.data_type['file_dir']  
        roi = self.data_type['roi']
        if file_dir == 'encoding':
            fpath = os.path.join(self.dirs.ENCODE_DIR, f's{self.subj:02}', f'Y_info_glm{self.glm}_{roi}.mat')
        elif file_dir == 'beta_roi':
            fpath = os.path.join(self.dirs.BETA_REG_DIR, f's{self.subj:02}', f'Y_info_glm{self.glm}_{roi}.mat')
        return io.read_mat_as_hdf5(fpath)['Y']
    
    def _add_task_conds(self):
        """ reads in `sc1_sc2_task_conds.txt` summary as dataframe
            takes the first `stim` from each group (i.e. takes one instance of `instruct`)
            Returns:
                pandas dataframe
        """
        constants = Defaults()
        json_dict = io.read_json(os.path.join(self.dirs.BASE_DIR, constants.conn_file))
        dataframe = pd.DataFrame.from_dict(json_dict) 
                
        exp_num = int((re.findall('\d+', self.exp))[0])  

        def strip(dataframe):
            for col in dataframe.columns:
                try: 
                    dataframe[col] = dataframe[col].str.strip()
                except: AttributeError
            return dataframe

        # remove trailing spaces from str cols
        dataframe = strip(dataframe)

        # filter dataframe
        if self.incl_inst:
            dataframe = dataframe.query(f'StudyNum=={exp_num}')
        else:
             dataframe = dataframe.query(f'StudyNum=={exp_num} and taskName!="Instruct"')

        return dataframe.groupby(f'{self.stim}Num', as_index = False).first().to_dict(orient = 'list')

    def _calculate_betas(self, X, Y):
        """ calculates weights (X^t*X)^-1*(X^tY)
            Args:
                X (array-like): shape (n_samples, n_features)
                Y (array-like): shape (n_features, n_targets)
            Returns:
                betas (array-like): shape (n_targets, n_samples)
        """
        # is this correct?
        # betas = np.linalg.pinv(X) @ Y[0:X.shape[0],:]
        betas = np.matmul(np.linalg.pinv(X), Y.T)

        return betas
    
    def _check_init(self):
        """ validates inputs for `glm` and `data_type`
        """
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
        """ set path to betas based on `roi` and `data_type`
            Returns: 
                fpath (dir): full path to beta file
        """
        # save dict to disk as HDF5 file obj
        roi = self.data_type['roi']
        if self.avg=='run':
            fname = f'mbeta_{roi}_all.h5'
        elif self.avg=='sess':
            fname = f'beta_{roi}_all.h5'

        if self.data_type['file_dir'] == 'encoding':
            fpath = os.path.join(self.dirs.ENCODE_DIR, fname)
        elif self.data_type['file_dir'] == 'beta_roi':
            fpath = os.path.join(self.dirs.BETA_REG_DIR, fname)
        
        return fpath

    def _concat_exps(self):
        """ preps betas and concats across experiments 
            Returns: 
                B_concat (nested dict): keys are exp
        """
        B_concat = {}
        for exp in self.experiment:

            # get directories for `exp`
            self.dirs = Dirs(study_name = exp, glm = self.glm)

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
            `sess` and `exp` and returns dict
            Args: 
                exps_dict (nested dict): nested dictionary with `exp` as key
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

# run the following to return data
# prep = DataManager()
# model_data = prep.get_conn_data()