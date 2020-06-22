#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:06:33 2020
Contains code for evaluation of different connectivity models

@author: ladan
"""
# import packages
import pandas as pd
import numpy as np
import scipy as sp
# import sklearn
import prep_data # module used to prepare data


# Define functions
def evaluate(M, subset = [], splitby = [], rois = ['grey_nan', 'tesselsWB162'], inclInst = 1,
             meanSubt = 0, experNum = [1, 2], glm = 7, avg = 1, crossed = 0):
    
    """
    evaluate will be used to evaluate connectivity models.
    Two types of evaluation can be used: crossed and not-crossed.
    
    INPUTS:
    - M        : the structure, dict or dataframe, containing the data for the model. Or outputs of sklearn
    - subset   : what data should the evaluation be based upon?
    - splitby  : by which variable should the evaluation be split?
    - rois     : a list of all the rois that you want to include in evaluation/modelling
    - inclInst : include the instructions in the analysis or not?
    - meanSubt : subtract the mean or not?
    - experNum : list with study id: 1 for sc1 and 2 for sc2. Do I need to include it as a variable?
    - glm      : glm number
    - avg      : average across runs within a session or not?!
    - crossed  : doubled crossed cross validation (set to 1), or not (set to 0)
    
    OUTPUTS:
    - Ypred : the predicted values for Y (cerebellar activity profiles)
    - Ytest : the Y array used for testing the model
    - R     : The structure with all the evaluation measures
    
        R will have:
            * Reliability of data 
            * Reliability of predictions
            * Sparseness measures (Gini index?)
            * Upper noise ceiling
            * Lower noise ceiling
            * RDMs with the predicted data! To calculate this use RDM function.
    """
    
    Y_roi = {} # a dictionary with the roi name as the key and Y as the value 
    for ri in np.arange(len(rois)):
        print('........ Doing get_wcon for %s' % rois[ri])
        # get data for all ROIs:
        print(Y_roi.keys())
        [Y_roi[rois[ri]], Sf] = prep_data.get_wcon(experNum = experNum, glm = glm, roi = rois[ri], avg = 1)
        print(Y_roi.keys())
        
    # Checking and creating subset and splitby
    ## subset:
    if not subset: #if subset is empty
        subset = (np.array(Sf['StudyNum'] == 2))*1 # an array with 0s for StudyNum = 1 and 1s for StudyNum = 2
        
    Sf['subset'] = subset # put subset into the dataframe (Sf)
    ## splitby:
    if not splitby: # if splitby is empty
        splitby = np.ones((subset.shape), dtype = int)
    Sf['splitby'] = splitby # put splitby into the dataframe (Sf)
    
    # getting the evaluation dataset
    sS     = Sf.loc[Sf.subset == 1]
    splits = np.unique(sS.splitby)
#     print(splits)

        
    
    
    #return (Ypred, Ytest, R)
    return (Y_roi, Sf)

def evaluate_all ():
    
    """
    Does the evaluation for all the models and subjects and returns data setructures ...
    which will be used for plotting.
    """
    
    return

