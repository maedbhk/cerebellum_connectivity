#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:08:47 2020

@author: ladan
"""
# import packages
import os # to handle path information
import pandas as pd
import numpy as np
import scipy as sp
import data_integration as di
import essentials as es
import pickle # used for saving data (pip/pip3 install pickle-mixin)
import prep_data

# setting some defaults paths
baseDir         = '/Users/ladan/Documents/Project-Cerebellum/Cerebellum_Data'
behavDir        = 'data'
imagingDir      = 'imaging_data'
suitDir         = 'suit'
regDir          = 'RegionOfInterest'
connDir         = 'connModels' # setting a different directory than in sc1sc2_connectivity
suitToolDir     = '/Users/ladan/Documents/MATLAB/suit'
encodeDir       = 'encoding'

# setting some default variables
#returnSubjs = np.array([2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31])

# define functions
def model(sn, method, params, glm = 7, rois = {'cortex':'tesselsWB162', 'cerebellum':'grey_nan'}, 
          trainMode = 'crossed', trainExper = 1, inclInstr = 1, scale = True, 
          overwrite = True, avg = 1):
    """
    model uses connect_fit module to fit models for different subjects using the options provided
    
    INPUTS:
    sn         : subjects you want to do the modelling for
    method     : method of regression you want to use for modelling: plsregress, olsregress, 
                 pcregress, ridgeregress
    params     : parameters of the model you need to set
    glm        : glm number
    rois       : contains strings with the names of the rois
    trainMode  : training mode: 'crossed' flipping session between X and Y, 'uncrossed': not flipping sessions
    trainExper : the study (experiment) you want to use for training, 1 (sc1) or 2 (sc2)
    inclInstr  : flag indicating whether you want to include instructions in the modelling or not
    scale      : scale data before modelling? 1 (yes) 0 (no)
    overwrite  : overwrite the existing data saved for modelling or not? 1 (yes) 0 (no)
    avg        : flag indicating whether data should be averaged across runs of each sessions or not! 1 (yes)
    
    OUTPUTS:
    RR         :
    """
    
        
    # Setting directories
    name     = 'mb4_%s_%s'% (rois['cortex'], method)
    outDir   = os.path.join(baseDir, 'sc%d'% trainExper, connDir, 'glm%d'%glm, name);
    
        
    
    # use prep_data.get_wcon to get the data
    Data = {} # dictionary that will have the roi names as its keys
    for ri in list(rois.keys()):
        [Data[ri], Tf] = prep_data.get_wcon(experNum = [1, 2], glm = 7, roi = rois[ri], avg = avg)
        
    Tf.to_csv(os.path.join(baseDir, 'test.csv'), index=False)
    X = Data['cortex']
    Y = Data['cerebellum']
        
        
    # Find the data that we want to use for fitting the connectivity
    SI1 = np.argwhere(np.array(((Tf['StudyNum'] == trainExper)*1)*((Tf['sess'] == 1)*1) == 1))
    SI2 = np.argwhere(np.array(((Tf['StudyNum'] == trainExper)*1)*((Tf['sess'] == 2)*1) == 1))
    
    # Arrange data based on the training mode
    trainXindx = np.concatenate((SI1, SI2))
    if trainMode == 'crossed':
        trainYindx = np.concatenate((SI2, SI1))
    elif trainMode == 'uncrossed':
        trainYindx = np.concatenate((SI1, SI2))
        
    trainXindx = trainXindx.flatten()
    trainYindx = trainYindx.flatten()
        
        
    # Estimate the model and store the information attached to it
    RR = {} # dictionary with all the info for the model
    for s in sn:
        print('........ Doing Modelling for s%02d'% s)
        outname = os.path.join(outDir, '%s_s%02d.dat'%(name, s))
        
        # add the new model to the previous one or over-write it?
        if (os.path.exists(outname) and overwrite == True):
            R = pickle.load(open(outname, "rb"))
        else:
            R = {}
            
        # Get data
        xx = X['s%02d'%s][trainXindx, :]
        yy = Y['s%02d'%s][trainYindx, :]


    return xx, yy, Tf, trainXindx, trainYindx