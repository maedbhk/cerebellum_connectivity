#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:48:47 2020
Used for preparing data for connectivity modelling and evaluation

@author: ladan
"""

# importing packages
import os # to handle path information
#import pandas as pd
import numpy as np
#import scipy as sp
import data_integration as di
import essentials as es
import pickle # used for saving data (pip/pip3 install picke-mixin)

# setting some default paths
baseDir         = '/Users/ladan/Documents/Project-Cerebellum/Cerebellum_Data'
behavDir        = 'data'
imagingDir      = 'imaging_data'
suitDir         = 'suit'
regDir          = 'RegionOfInterest'
connDir         = 'connModels' # setting a different directory than in sc1sc2_connectivity
suitToolDir     = '/Users/ladan/Documents/MATLAB/suit'
encodeDir       = 'encoding'

# sestting defaults for some variables
returnSubjs = np.array([2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31])


# define functions
def get_data(sn = returnSubjs, glm = 7, roi = 'grey_nan', which = 'cond', avg = 1):
    """
    get_data prepares per subject data for modelling
    INPUTS
    sn    : a numpy array with ids of all the subjects that will be used 
    roi   : string specifying the name of the ROI you want the data for
    glm   : a number repesenting the glm Number you want to use: 7 (conditions), or 8 (tasks)
    which : will be used in creating the indicator matrix. can be 'task' or 'cond'
    avg   : flag indicating whether you want to average across runs within a session or not
    
    OUTPUTS
    data_dict:
    
    
    Some additional notes:
    The data type for the variable that stores all the data should be decided. 
    For now, I'm gonna go with nested dictionaries:
    It can go two ways:
    1. B_dict{experiment_num:{subject:{session:B_sess} with which we will end up with 
    a single dictionary for both experiments which can be saved. This is returned by the function
    2. B_dict{subject:{session:B_sess}} with which we will end up with one dictionary for each experiment, 
    each saved in a directory specific to that experiment. This is the way the matlab code was saving the variable.
    
    Prepared data will be saved using (pickle: pip/pip3 install pickle-mixin) as binary files with .dat format
    To load these saved variables later: 
        pickle.load(open(filename, "rb")) # "rb" stands for Reading Binary file

    """
    # looping over experiments
    data_dict = {} # will store all the data
    for e in np.arange(1, 3):
        print('preparing data for study %d'% e)

        # setting directories
        glmDir      = os.path.join(baseDir , 'sc%d'% e , 'GLM_firstlevel_%d'% glm)
        encodingDir = os.path.join(baseDir , 'sc%d'% e , encodeDir , 'glm%d'% glm)
        
        B_alls = {} # data for each experiment will be saved in a dictionary
        for s in sn:
            print('Doing subject %02d' % s)

            # getting the data for the roi
            path2data = os.path.join(encodingDir , 's%02d'%s ,'Y_info_glm%d_%s.mat' %(glm, roi))
            data_dict = di.matImport(path2data, form = 'dict') # import data as a dictionary

            D = data_dict['data']

            # getting SPM_info for the tasks info
            path2spm = os.path.join(glmDir , 's%02d'%s , 'SPM_info.mat')
            T        = di.matImport(path2spm, form = 'dict') # import data as a dictionary
            
            ## for now it assumes that you want to average subtract the average across runs of a session
            # Now generate mean estimates per session
            # Note that in the new SPM_info Instruction is coded as cond =0
            T['cond'] = T['cond'] + 1   # This is only for generating the avergaging matrix
                
            B_sess = {}
            for se in [1, 2]:
                if avg == 1:
                    X          = es.indicatorMatrix('identity_p', T[which]*(T['sess'] == se))

                    # np.linalg.pinv is pretty slow!
                    # also, we need to decide on an appropriate data type here
                    ## I'm gonna go with nested dictionaries here!
                    B_sess[se]   = np.linalg.pinv(X) @ D[0:X.shape[0],:] # This calculates average beta across runs, skipping intercepts, for each session
                    
                #elif avg == 0: # UNDER "CONSTRUCTION"
                
            B_alls[s] = B_sess
    
        data_dict[e] = B_alls
        #data_dict['avg'] =
        
        outfile = os.path.join(encodingDir, 'mbeta_%s_all.dat'% roi)
        pickle.dump(B_alls, open(outfile, "wb")) # "wb": Writing Binary file
    
    return data_dict