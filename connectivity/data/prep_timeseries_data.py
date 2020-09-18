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
        """ prepares data for modelling and evaluation
        pulls data from imaging data directly for use in time series modelling.
        Returns:
            T_all (nested dict):
        """"
        # check that we're setting the correct parameters
        self._check_init()
        
        # return `exp` data
        data_dict = self._concat_exps()
        
        # mask data
        
        # delay data
        
        # return concatenated info 
        
       
    
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
            fname = 's%s/rrun_%s.nii'
        if self.data_type['file_dir'] == 'imaging_data':
            fpath = os.path.join(self.dirs.IMAGING_DIR, fname)
        return fpath
            
    
    def _concat_exps(self):
        """ retrieves data:
        Returns:
            T_concat(dict): keys are exp - values are data in shape (time, x,y,z (48, 84, 84))
        """
        
        T_concat = dict()
        
        for self.subj in self.subjects:
            sub_concat = dict()
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
                # load imaging data from nii
                data_runs = []
                for run in runs:
                    data_runs.append(nib.load(fpath%(self.subj, run)).get_data().T)
                self_concat[exp] = np.concatenate(data_runs)
            T_concat[self.subj] = self_concat
            
        return T_concat
    
    def _get_masks(self):
        """ return cerebellar and cortical masks for each subject minus the buffer voxels.
        
        Returns:
           masks (nested dict): keys are subjects followed by keys as "cerebellum" and "cortex"
        """
        
        masks = dict()
        self.dirs = Dirs(study_name=self.exp, glm=self.glm)
        for self.subj in self.subjects:
            individ_masks = dict()
            fname = f'{self.subj}/maskbrainSUITGrey.nii
            fpath = os.path.join(self.dirs.SUIT_ANAT_DIR, fname)
            
            cerebellar = nib.load(fpath).get_data().T
            
            fname = f'{self.subj}/rmask_gray.nii'
            fpath = os.path.join(self.dirs.IMAGING_DIR, fname)
            cortex = nib.load(fpath).get_data().T
            
            fname = f'{self.subj}/buffer_voxels.nii'
            fpath = os.path.join(self.dirs.SUIT_ANAT_DIR, fname)
            buffer = nib.load(fpath).get_data().T
            
            cerebellar[buffer!=0]=0
            cerebellar[cerebellar!=0]=1
            cortex[buffer!=0]=0
            cortex[cerebellar!=0]=0
            cortex[cortex!=0]=1
            
            individ_masks['cerebellum'] = cerebellar.astype('bool')
            individ_masks['cortex'] = cortex,astype('bool')
            
            masks[f'{self.subj}'] = individ_masks
         
        
        return masks
    
    def _check_init(self):
        """ validates inputs for 'data_type' and 'glm'
        """
        
        if self.glm == 7:
            self.stim = 'cond'
        elif self.glm ==8:
            self.stim='task'
        elif self.glm == 'none':
            self.stim = 'timeseries'
        else:
            print('choose a valid glm')
        
        roi = self.data_type['roi']
        if roi == 'cerebellum_grey':
            self.data_type['file_dir'] = 'beta_roi'
        elif roi == 'grey_nan':
            self.data_type['file_dir'] = 'encoding'
        elif roi == 'voxelwise':
            self.data_type['file_dir'] = 'imaging_data'
        
            
    
    
        