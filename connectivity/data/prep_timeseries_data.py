import os
import numpy as np
import re
import deepdish as dd
import copy
from collections import defaultdict
import nibabel as nib
from connectivity.constants import Defaults, Dirs
from connectivity import io
from connectivity.helper_functions import AutoVivification
from connectivity.indicatormatrix import indicatorMatrix

"""
Created on Fri September 11 9:19:00 2020
Prepares times series data for connectivity modelling and evaluation

@authors: Amanda LeBel, Ladan Shahshahani, and Maedbh Kind

"""

class DataManager:
    """ Data manager class, preps preproccesed time series data for connectivity
    modelling. Initialises inputs for DataManager Class:
        experiment (list): default is ['sc1', and 'sc2']
        sessions (list): default is [1, 2]
        data_type (dict): default is {'roi': 'voxelwise', 'file_dir': 'imaging_data'}

        number_of_delays(int): default is 3. Value must be positive. 
        
        """
    
    def __init__(self):
        self.experiment = ['sc1', 'sc2']
        self.sessions = [1, 2]
        self.data_type = {'roi': 'voxelwise', 'file_dir': 'imaging_data'}
        self.number_of_delays = 3
        
    def get_conn_data(self):
        pass
    
    def make_delayed(self, arr, delays):
        """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
        (in samples). This is effectively a finite impulse response function to model the hemodynamic 
        response function.
        
        Args:
            arr (array like): shape [n_timepoints, n_voxels]
            delays (list like): list of delay values to use. Default range(-1, 4, 1)
           
        """
        nt,ndim = arr.shape
        dstims = []
        for di,d in enumerate(delays):
            dstim = np.zeros((nt, ndim))
            if d<0: ## negative delay
                dstim[:d,:] = arr[-d:,:]
            elif d>0:
                dstim[d:,:] = arr[:-d,:]
            else: ## d==0
                dstim = arr.copy()
            dstims.append(dstim)
        return np.hstack(dstims)
    
    def _get_path_to_data(self):
        """ Set path to data based on 'roi' and and 'data_type'
        returns:
        fpath(dir): full path to data file
        """
        roi = self.data_dtype['roi']
        if roi == 'voxelwise':
            fname = f'rrun_%s.nii'
        if self.data_type['file_dir'] == 'imaging_data':
            fpath = os.path.join(self.dirs.IMAGING_DIR, fname)
        return fpath
            
    
    def _concat_exps(self):
        """ retrieves data:
        Returns:
            T_concat(nested dict): keys are exp
        """
        
        T_concat = {}
        for exp in self.experiment:
            
            # Get directories for 'exp'
            self.dirs = Dirs(study_name=exp, glm=self.glm)
            
            # load data filepaths for 'exp'
            fpath = self._get_path_to_data()
            
            # get runs for data
            if exp == 'sc1':
                runs = list(range(1, 16, 1))
            elif exp == 'sc2':
                runs = list(range(16, 33, 1))
            for run in runs:
                
                        
            # load imaging data from nii
            
        return T_concat
    
    def _get_masks(self):
        pass
    
    
        