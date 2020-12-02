# import libraries and packages
import os 
import pandas as pd
import numpy as np
import re
import deepdish as dd
import copy
from collections import defaultdict
import scipy.io as sio 
import h5py
from sklearn.preprocessing import StandardScaler

# Import module as such - no need to make them a class
from connectivity.constants import Defaults, Dirs
import connectivity.io as cio  
from connectivity.helper_functions import AutoVivification
from connectivity.indicatormatrix import indicatorMatrix

"""
Created on Fri Jun 19 11:48:47 2020
Prepares data for connectivity modelling and evaluation

@authors: Ladan Shahshahani, Maedbh King., and Jörn Diedrichsen 
"""

class Dataset: 
    """ Dataset class, holds betas for one region, one experiment, one subject for connectivity modelling
    """

    def __init__(self,experiment = 'sc1',glm = 7, roi = 'cerebellum_grey', 
                 sn = 3):
        self.exp = experiment
        self.glm = glm
        self.roi = roi 
        self.sn = sn 
        self.data = None

    def import_mat(self): 
        """ Reads a data set from the Y_info file and corresponding GLM file from matlab 
        """
        dirs = Dirs(study_name = self.exp, glm = self.glm)
        fname =  'Y_info_' + f'glm{self.glm}' + '_' + self.roi + '.mat'
        fdir = dirs.BETA_REG_DIR / dirs.BETA_REG_DIR / f's{self.sn:02}'
        file = h5py.File(fdir / fname,'r')
        # Store the data in betas x voxel/rois format 
        self.data = np.array(file['Y']['data']).T
        # this is the row info 
        self.info = cio.convert_to_dataframe(file['Y'],['CN','SN','TN','cond','inst','run','sess','task'])
        return self

    def save(self,filename = None): 
        """ Save the content of the data set in a dict as a hpf5 file
        """
        if filename is None: 
            dirs = Dirs(study_name = self.exp, glm = self.glm)
            fname =  'Y_info_' + f'glm{self.glm}' + '_' + self.roi + '.h5'
            fdir = dirs.BETA_REG_DIR / dirs.BETA_REG_DIR / f's{self.sn:02}'

        dd.io.save(fdir / fname, self, compression = None)

    def load(self,filename = None):
        """ Load the content of a data set object from a hpf5 file 
        """ 
        if filename is None: 
            dirs = Dirs(study_name = self.exp, glm = self.glm)
            fname =  'Y_info_' + f'glm{self.glm}' + '_' + self.roi + '.h5'
            fdir = dirs.BETA_REG_DIR / dirs.BETA_REG_DIR / f's{self.sn:02}'

        dd.io.save(fdir / fname, self, compression = None)
