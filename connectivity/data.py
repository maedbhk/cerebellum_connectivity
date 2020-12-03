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
import connectivity.matrix as matrix
from numpy.linalg import solve

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

    def load_mat(self): 
        """ Reads a data set from the Y_info file and corresponding GLM file from matlab 
        """
        dirs = Dirs(study_name = self.exp, glm = self.glm)
        fname =  'Y_' + f'glm{self.glm}' + '_' + self.roi + '.mat'
        fdir = dirs.BETA_REG_DIR / dirs.BETA_REG_DIR / f's{self.sn:02}'
        file = h5py.File(fdir / fname,'r')
        # Store the data in betas x voxel/rois format 
        self.data = np.array(file['data']).T
        # this is the row info 
        self.XX = np.array(file['XX'])
        self.CN = cio._convertobj(file,'CN')
        self.TN = cio._convertobj(file,'CN')
        self.cond = np.array(file['cond']).reshape(-1).astype(int)
        self.inst = np.array(file['inst']).reshape(-1).astype(int)
        self.task = np.array(file['task']).reshape(-1).astype(int)
        self.sess = np.array(file['sess']).reshape(-1).astype(int)
        self.run = np.array(file['run']).reshape(-1).astype(int)
        return self

    def save(self,filename = None): 
        """ Save the content of the data set in a dict as a hpf5 file
        """
        if filename is None: 
            dirs = Dirs(study_name = self.exp, glm = self.glm)
            fname =  'Y_' + f'glm{self.glm}' + '_' + self.roi + '.h5'
            fdir = dirs.BETA_REG_DIR / dirs.BETA_REG_DIR / f's{self.sn:02}'

        dd.io.save(fdir / fname, vars(self), compression = None)

    def load(self,filename = None):
        """ Load the content of a data set object from a hpf5 file 
        """ 
        if filename is None: 
            dirs = Dirs(study_name = self.exp, glm = self.glm)
            fname =  'Y_info_' + f'glm{self.glm}' + '_' + self.roi + '.h5'
            fdir = dirs.BETA_REG_DIR / dirs.BETA_REG_DIR / f's{self.sn:02}'

        dd.io.load(fdir / fname, self, compression = None)

    def get_info(self): 
        d = {'CN':self.CN,'TN':self.TN,'sess':self.sess,'run':self.run,'inst':self.inst,'task':self.task,'cond':self.cond}
        return pd.DataFrame(d)

    def get_data(self, averaging = 'sess', weighting = True, instr = True):
        """ Get the data using a specific aggregation 
            Returns it as a numpy array and the information as a Pandas data frame
            Parameters: 
                averaging (str)
                    'sess': within each session
                    'none': no averaging 
                    'exp': across the whole experiment
                weighting (bool)
                    Should the betas be weighted by X.T * X  
                instr (bool)
                    Include instruction regressors? 
            Returns: 
                data (np.array)
                    Aggregated data 
                info (pd.Dataframe)
                    information for the aggregated data 
        """ 
        num_runs = max(self.run)
        num_reg = sum(self.run==1)
        N = self.sess.shape[0]
        T = self.get_info() 
        # Regressor ID: this is assuming same structure across all runs! 
        T['id'] = np.kron(np.ones((num_runs,)),np.arange(num_reg)).astype(int)
        # Weighting all regressors by XX 
        if weighting:
            self.weight = np.zeros((N,))
            for r in range(num_runs):
                idx = (self.run ==r+1)
                self.weight[idx,] = np.sqrt(np.diag(self.XX[r,:,:]))
            Y = self.data * self.weight.reshape(-1,1)
        else:
            Y = self.data
        # Different ways of averaging 
        if (averaging == 'sess'):
            X = matrix.indicator(T.id + (self.sess-1) * num_reg)
            Y = np.linalg.solve(X.T @ X, X.T @ Y)
            S = T[np.logical_or(T.run==1,T.run==9)]
        elif (averaging == 'exp'):
            X = matrix.indicator(T.id)
            Y = np.linalg.solve(X.T @ X, X.T @ Y)
            S = T[T.run==1]
        elif (averaging == 'none'):
            S = T 
        else:
            raise(NameError('averaging needs to be sess, exp, or none'))
        return Y,S