import os
import numpy as np
import re
import deepdish as dd
import h5py
import copy
from collections import defaultdict
import nibabel as nib
import csv
from connectivity.constants import Defaults, Dirs
from connectivity import io
from connectivity.helper_functions import AutoVivification
from connectivity.indicatormatrix import indicatorMatrix
import connectivity.data.detrend as detrend

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
        self.stim = 'timeseries'
        self.sessions = [1, 2]
        self.glm = 'none'
        self.data_type = {'roi': 'voxelwise', 'file_dir': 'imaging_data'}
        self.number_of_delays = 3
        self.subjects = [6]
        self.detrend = 'sg' #options are 'sg' and 'lin'
        self.structure = ['cerebellum', 'cortex']
        self.redundancy = True
  
        
    def get_conn_data(self):
        """ prepares data for modelling and evaluation
        pulls data from imaging data directly for use in time series modelling.
        Returns:
            T_all (nested dict): structure of dict below
                +--betas
                |   +--cerebellum_undelayed
                |      +--Subj
                |         +--sc1
                |         |   +--ses1
                |         |   +--ses2
                |         +-sc2
                |             +--ses1
                |             +--ses2
                |   +--cerebellum_delayed
                |      +--Subj
                |         +--sc1
                |         |   +--ses1
                |         |   +--ses2
                |         +-sc2
                |             +--ses1
                |             +--ses2
                |   +--cortex_delayed
                |      +--Subj
                |         +--sc1
                |         |   +--ses1
                |         |   +--ses2
                |         +-sc2
                |             +--ses1
                |             +--ses2
                |   +--cortex_undelayed
                |      +--Subj
                |         +--sc1
                |         |   +--ses1
                |         |   +--ses2
                |         +-sc2
                |             +--ses1
                |             +--ses2
                +--masks
                   +--Subj
                      +--cortical mask
                      +-- cerebellar mask
                   
        """
        # check that we're setting the correct parameters
        self._check_init()
        
        # return `exp` data
        data_dict = self._concat_exps()
        # initializes temporary dict for holding reorganized data
        temp_dict = dict()
        # return mask information
        masks = self._get_masks()
        
                        
        
        # temporally detrend data
        for self.subj in self.subjects:
            for self.exp in self.experiment:
                raw_data = data_dict[f's{self.subj:02}'][f'{self.exp}']
                
                
                detrend_data = [self._detrend_data(d) for d in raw_data]
                
                for self.sess in self.sessions:
                    if self.sess == 1:
                        r = list(range(0, 8))
                    elif self.sess ==2:
                        r = list(range(8,17))  
                    if self.redundancy == True:
                        concat_detrend_data = self._remove_redundant_tasks(detrend_data[r[0]:r[-1]], self.exp, self.subj, self.sess)
                        print(f'redundant tasks removed from data set. New shape of data is: {raw_data.shape}') 
                    else:
                        concat_detrend_data = np.concatenate(detrend_data[r[0]:r[-1]], axis=0)
                        print(f'Detrended data for sub: {self.subj} and exp: {self.exp} and sess: {self.sess} is shape {np.concatenate(detrend_data, axis=0).shape}')

                    # mask data
                    for struct in self.structure:
                        masked_data = concat_detrend_data[:, masks[f's{self.subj:02}'][struct]]
                        print(f'masked data is of shape:{masked_data.shape}')
                        # delay data (to account for hemodynamic response variation
                        if self.number_of_delays !=0:
                            delays = range(-1, self.number_of_delays-1)
                            delayed_data = self.make_delayed(masked_data, delays)
                        else:
                            print('Data is not being delayed. This is not recommended for best performance.')
                            delayed_data = masked_data
                        print(f'Delayed data is of shape: {delayed_data.shape}')
                        # change the nesting order of the dictionary
                        if f'{struct}_delayed' in temp_dict:
                            temp_dict[f'{struct}_delayed'][f'{self.exp}']['betas'][f's{self.subj:02}'].update({str(self.sess):delayed_data})
                        else:
                            temp_dict[f'{struct}_delayed'] = {f'{self.exp}':{'betas':{f's{self.subj:02}':{f'{self.sess}': delayed_data}}}}
                        if  f'{struct}_undelayed' in temp_dict:
                            temp_dict[f'{struct}_undelayed'][f'{self.exp}']['betas'][f's{self.subj:02}'].update({str(self.sess):masked_data})
                        else:
                            temp_dict[f'{struct}_undelayed'] = {f'{self.exp}':{'betas':{f's{self.subj:02}':{f'{self.sess}': masked_data}}}}
                        
      
        # return concatenated info 
        T_all = dict()
        T_all['betas'] = temp_dict
        T_all['masks'] = masks
        return T_all
       
    
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
        roi = self.data_type['roi']
        
        if roi == 'voxelwise':
            fname = f's{self.subj:02}/rrun_%02d.nii'
        if self.data_type['file_dir'] == 'imaging_data':
            fpath = os.path.join(self.dirs.IMAGING_DIR, fname)
        return fpath
            
    def _get_masks(self):
        """ return cerebellar and cortical masks for each subject minus the buffer voxels.

        Returns:
        masks (nested dict): keys are subjects followed by keys as "cerebellum" and "cortex"
        """

        masks = dict()
        self.dirs = Dirs(study_name='sc1', glm=7)
        for self.subj in self.subjects:
            individ_masks = dict()
            fname = f's{self.subj:02}/maskbrainSUITGrey.nii'
            fpath = os.path.join(self.dirs.SUIT_ANAT_DIR, fname)

            cerebellar = nib.load(fpath).get_data().T

            fname = f's{self.subj:02}/rmask_gray.nii'
            fpath = os.path.join(self.dirs.IMAGING_DIR, fname)
            cortex = nib.load(fpath).get_data().T

            fname = f's{self.subj:02}/buffer_voxels.nii'
            fpath = os.path.join(self.dirs.SUIT_ANAT_DIR, fname)
            buffer = nib.load(fpath).get_data().T

            cerebellar[buffer!=0]=0
            cerebellar[cerebellar!=0]=1
            cortex[buffer!=0]=0
            cortex[cerebellar!=0]=0
            cortex[cortex!=0]=1

            individ_masks['cerebellum'] = cerebellar.astype('bool')
            individ_masks['cortex'] = cortex.astype('bool')

        masks[f's{self.subj:02}'] = individ_masks


        return masks
    
    def _concat_exps(self):
        """ retrieves data:
        Returns:
            T_concat(dict): keys are exp - values are data in shape (scan, time, x,y,z (48, 84, 84))
        """
        
        T_concat = dict()
        
        for self.subj in self.subjects:
            sub_concat = dict()
                
                
            for exp in self.experiment:
                print(f'retrieving data for s{self.subj:02} ...')
                # Get directories for 'exp'
                self.dirs = Dirs(study_name='sc1', glm=7)
                if exp == 'sc1':
                        d = 'exp1'
                    elif exp == 'sc2':
                        d = 'exp2'
                try:
                    assert self.data_type['file_dir'] == 'imaging_data'
                    assert self.data_type['roi'] == 'voxelwise'
                    
                    sub_concat[exp] = dd.io.load(os.path.join(self.dirs.IMAGING_DIR, f's{self.subj:02}/rrun_{exp}.hf5'))[d]
                except:
                    print('Data not found in HDF5, loading from nifti...')
                    # load data filepaths for 'exp'
                    fpath = self._get_path_to_data()

                    # get runs for data
                    if exp == 'sc1':
                        runs = list(range(1, 17, 1))
                    elif exp == 'sc2':
                        runs = list(range(17, 33, 1))
                    # load imaging data from nii
                    filenames= [fpath%(run) for run in runs]
                    data_runs = [nib.load(filename).get_data().T for filename in filenames]
                    dd.io.save(os.path.join(self.dirs.IMAGING_DIR, f's{self.subj:02}/rrun_{exp}.hf5'), {d: data_runs})
                    sub_concat[exp] = data_runs
            T_concat[f's{self.subj:02}'] = sub_concat
            
        return T_concat
    
    def _remove_redundant_tasks(self, arr, exp, subj, sess):
        
        """ remove tasks that are in both the train set and the evaluate set. This will help prevent excessive overfitting
        
        Parameters:
            arr (array): 4d array with first dimension of time(TR) and last three dimensions are x,y,z
        Returns:
            arr (array): 4d array; same as input missing the TRs with the redundant tasks
        """
        #task names for tasks in both sc1 and sc2
        tasks_A = ['GoNoGo', 'emotional', 'affective', 'arithmetic', 
           'intervalTiming', 'nBack', 'stroop', 'motorImagery', 'checkerBoard']
        
        tasks_B = ['spatialMap', 'mentalRotation', 'biologicalmotion', 'CPRO', 
          'prediction', 'respAlt', 'natureMovie', 
           'landscapeMovie', 'emotionProcess', 'romanceMovie']
        self.dirs = Dirs(study_name=exp, glm=7)
        new_data = dict()
        for i in range(0, np.shape(arr)[0]):
            temp_data = arr[i]
            if sess==1 and exp =='sc1':
                fname = os.path.join(self.dirs.TIME_DIR, f's{subj:02}/ses-a1/run-{i+1}_events.tsv')
            elif sess==1  and exp =='sc2':
                fname = os.path.join(self.dirs.TIME_DIR, f's{subj:02}/ses-b1/run-{i+1}_events.tsv')
            elif sess==2 and exp =='sc1':
                fname = os.path.join(self.dirs.TIME_DIR, f's{subj:02}/ses-a2/run-{i+1}_events.tsv')
            elif sess=2 and exp =='sc2':
                fname = os.path.join(self.dirs.TIME_DIR, f's{subj:02}/ses-b2/run-{i+1}_events.tsv')

            with open(fname) as tsvfile:
                reader = csv.DictReader(tsvfile, dialect='excel-tab')
                for row in reader:
                    if row['startTime'] != 'n/a':
                        x = temp_data[int(float(row['startTime'])): int(float(row['startTime'])+ float(row['duration']))]
                        if x.ndim == 4:
                            if row['taskName'] not in new_data:
                                new_data[row['taskName']] = x.tolist()
                            new_data[row['taskName']].append(x.tolist())
        for task in new_data.keys():
            print(task)
            temp = [np.array(i) for i in new_data[task] if np.array(i).ndim ==4]
            temp = np.concatenate(temp)
            new_data[task] = temp
        if exp == 'sc1':
            concat_data = np.concatenate([new_data[task] for task in tasks_A])
        if exp == 'sc2':
            concat_data = np.concatenate([new_data[task] for task in tasks_B])

        
        return concat_data
                            
  
    def _detrend_data(self, arr):
        """ 
        temporaly detrends data. Input should be from single scan for sg detrending
        
        Parameters:
            arr (array): 4d array with first dimension of time(TR) and last three dimensions are x,y,z
            
        Returns:
            arr(array): 4d array; same as input parameter
        """
        if self.detrend == 'sg':
            detrend_data = detrend.sgolay_filter_volume(arr, filtlen=121, degree=3)
            return detrend_data
        else:
            raise ValueError('This method of detrending is not yet supported')
    
    def _check_init(self):
        """ validates inputs for 'data_type' and 'glm'
        """
        assert self.stim == 'timeseries'
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
      
    

        
            
    
# run the following to return data
# prep = DataManager()
# model_data = prep.get_conn_data()  
