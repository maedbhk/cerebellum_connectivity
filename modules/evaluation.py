#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:34:09 2020
Contains function for evaluating models

@author: ladan
"""

# import packages
import os
import pickle # pip/pip3 install pickle-mixin
#import pandas as pd
import numpy as np
#import scipy as sp
# import sklearn
import prep_data # module used to prepare data
from sklearn.preprocessing import StandardScaler # for scaling the X

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
def evaluate_model(Md, subset = [], splitby = [], rois = {'cortex':'tesselsWB162', 'cerebellum':'grey_nan'},
             inclInst = 1, meanSubt = 1, experNum = [1, 2], glm = 7, avg = 1, crossed = 0):
    
    """
    evaluate will be used to evaluate connectivity models.
    Two types of evaluation can be used: crossed and not-crossed.
    
    INPUTS:
    - Md        : the structure, dict or dataframe, containing the data for the model. Or outputs of sklearn
    - subset    : what data should the evaluation be based upon?
    - splitby   : by which variable should the evaluation be split?
    - rois      : a dictionary of all the rois you want to include in the model
    - inclInst  : include the instructions in the analysis or not?
    - meanSubt  : subtract the mean or not?
    - experNum  : list with study id: 1 for sc1 and 2 for sc2. Do I need to include it as a variable?
    - glm       : glm number
    - avg       : average across runs within a session or not?!
    - trainMode : doubled crossed cross validation 'crossed', or not ('uncrossed')
    
    OUTPUTS:
    - Ypred : the predicted values for Y (cerebellar activity profiles)
    - Ytest : the Y array used for testing the model
    - R     : The structure with all the evaluation measures
    
        R will have:
            * Reliability of data 
            * Reliability of predictions
            * Sparseness measures (Gini index?)
            * double cross validated correlation
            * not double cross validated correlation
    """
    
    Z_roi = {} # a dictionary with the roi name as the key and X and Y arrays as the value 
    for ri in list(rois.keys()):
        print('........ Doing get_wcon for %s' % rois[ri])
        # get data for all ROIs:
        [Z_roi[rois[ri]], Sf] = prep_data.get_wcon(experNum = experNum, glm = glm, roi = rois[ri], avg = 1)
        
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
    Ss     = Sf.loc[Sf.subset == 1]
    splits = np.unique(Ss.splitby)
    
    # Loop over models and evaluate
    numModels = len(Md['params'])
    
    # define a dictionary with the keys and empty values:
    RR = {'sn':[], 'params':[], 'model':[], 'trainMode':[], 'xname':[],
         'Rcv':[], 'Rnc':[], 'Ry':[], 'Rp':[],
         'splits':[], 
         'spIdx':[], 'ginni':[]}
    
    for im in np.arange(0, numModels):
        print('........ Evaluating %s with param %s for s%02d '% 
              (Md['model'][im], Md['params'][im], Md['sn'][im]))
        
        # getting the model with the specified parameter
        Mlocal = Md['M'][im]
        
        # getting the indices of good estimations for evaluation
        Y_roi = Z_roi[rois['cerebellum']]['s%02d'%Md['sn'][im]]
        X_roi = Z_roi[rois['cortex']]['s%02d'%Md['sn'][im]]
        
        goodindx = (np.sum(abs(Y_roi), axis = 0)>0) * ((np.isnan(np.sum(Y_roi, axis = 0))*1) == 0) * ((np.isnan(np.sum(Mlocal['W'], axis = 1)))*1 == 0)
                
        for spl in np.arange(0, len(splits)):
                if meanSubt == 1:
                    for sess in [1, 2]:
                        arr = ((Sf.subset == 1)*1)*((Sf.sess == sess)*1)*(Sf.splitby == splits[spl]*1)
                        indx = np.argwhere(np.array(arr) == 1) # the input to np.argwhere is converted to array cause np.argwhere does not work with pandas series
                        
                        
                        Y_roi[indx,:] = Y_roi[indx,:] - np.mean(Y_roi[indx,:], axis = 0)
                        X_roi[indx,:] = X_roi[indx,:] - np.mean(X_roi[indx,:], axis = 0)
                        

                # Make one index that arranges the data [sess1;sess2]
                # Make one index that arranges the data [sess2;sess1]
                sess1 = np.argwhere(np.array(((Sf.subset == 1)*1)*((Sf.sess == 1)*1)*(Sf.splitby == splits[spl])))
                sess2 = np.argwhere(np.array(((Sf.subset == 1)*1)*((Sf.sess == 2)*1)*(Sf.splitby == splits[spl])))

                testAindx = (np.concatenate((sess1, sess2), axis = 0)).flatten()
                testBindx = (np.concatenate((sess2, sess1), axis = 0)).flatten()
                
                # Get the predicted values
                ## using models estimated with sklearn, I can use reg key from Mlocal
                #???????????????????????????????????????????????????????????????????????
                scalerZ = StandardScaler()
                scalerZ.fit(X_roi[testAindx, :])
                XtestA  = scalerZ.transform(X_roi[testAindx, :])
                
                scalerZ = StandardScaler()
                scalerZ.fit(X_roi[testBindx, :])
                XtestB  = scalerZ.transform(X_roi[testBindx, :])
                #???????????????????????????????????????????????????????????????????????
                
                predY   = Mlocal['reg'].predict(XtestA) # Predicted Y using crossvalidation
                predYnc = Mlocal['reg'].predict(XtestB) # Predicted Y not crossvalidated


                # Caluculate the respective sums-of-squares 
                Bindxd     = Y_roi[testBindx,:];
                Y_roi_good = Bindxd[:, goodindx]
                
                SSP   = sum(sum(predY[:,goodindx]**2))                   # Sum of square of predictions
                SSY   = sum(sum(Y_roi[testBindx,:]**2))                  # Sum of squares of data
                SSCp  = sum(sum(predY[:,goodindx]*predYnc[:,goodindx]))  # Covariance of Predictions
                SSCy  = sum(sum(Y_roi_good*Y_roi_good))                  # Covariance of Y's
                SSCn  = sum(sum(predYnc[:,goodindx]*Y_roi_good))         # Covariance of non-cross prediction and data
                SSCc  = sum(sum(predY[:,goodindx]*Y_roi_good))           # Covariance of cross prediction and data
                
                RR['sn'].append(Md['sn'][im])
                RR['params'].append(Md['params'][im])
                RR['model'].append(Md['model'][im])
                RR['trainMode'].append(Md['trainMode'][im])
                RR['xname'].append(rois['cortex'])
                
                RR['Rcv'].append(SSCc / np.sqrt(SSY*SSP)) # Double-Crossvalidated predictive correlation
                print(f"Rcv is {RR['Rcv']}")
                RR['Rnc'].append(SSCn / np.sqrt(SSY*SSP)) # Not double-crossvalidated predictive correlation
                print(f"Rnc is {RR['Rnc']}")
                
                # If we knew the true pattern, the best correlation we
                # would expect is sqrt(Ry) 
                # We also want to take into account the relaibility 
                RR['Ry'].append(SSCy / SSY)               # Reliability of data
                print(f"Ry is {RR['Ry']}")
                RR['Rp'].append(SSCp / SSP)               # Reliability of prediction
                print(f"Rp is {RR['Rp']}")
                RR['splits'].append(splits[spl])
                                          
                # Calucate Sparseness measures??????????
                Ws         = np.sort(np.abs((Mlocal['W'].transpose())), axis = 0) # transpose to match the shapes of matrices in matlab
                Wss        = Ws/np.sum(Ws, axis = 0)                              # standardized coefficients (to the sum overall
                RR['spIdx'].append(np.nanmean(Wss[-1,:]))                         # Largest over the sum of the others
                
                N = Wss.shape[0]
                w = (N - np.arange(1, N+1)+0.5)/N

                ginni      = 1 - 2*sum((Wss.transpose()*w).transpose())
                RR['ginni'].append(np.nanmean(ginni))                             # Ginni index: mean over voxels
# # #                 R.numReg = M.numReg(m); %% ?????????????????????
                    
    
    return (predY, Y_roi[testBindx,:], RR)

def evaluate_pipeline(sn, model, glm = 7, subset = [], splitby = [], rois = {'cortex':'tesselsWB162', 'cerebellum':'grey_nan'},
             inclInst = 1, meanSubt = 1, experNum = [1, 2], avg = 1, trainMode = 'crossed', trainExper = 1):
    """
    loads all the fitted models for the subjects entered in the array for sn, does the evaluation
    and saves the evaliation measures.
    INPUTS
    - sn       : subjects
    - model    : the model you want to evaluate. You can enter multiple models in a list.
                 Always give model as a list!
    - subset   : what data should the evaluation be based upon?
    - splitby  : by which variable should the evaluation be split?
    - rois     : a dictionary of all the rois you want to include in the model
    - inclInst : include the instructions in the analysis or not?
    - meanSubt : subtract the mean or not?
    - experNum : list with study id: 1 for sc1 and 2 for sc2. Do I need to include it as a variable?
    - glm      : glm number
    - avg      : average across runs within a session or not?!
    - crossed  : doubled crossed cross validation (set to 1), or not (set to 0)
    
    OUTPUTS
    - ER       : a dictionary containing evaluation parameters for all the subjects with subject ids as keys
    
    """
    
    # setting directories
    for ms in model: # ms: model string
        
        modelName = 'mb4_%s_%s'% (rois['cortex'], ms)
        modelDir  = os.path.join(baseDir, 'sc%d'% trainExper, connDir, 'glm%d'%glm, modelName)

        # get the testExper
        testExper = [item != trainExper for item in experNum] # get the test Experiment
        testDir   = 'sc%d'% np.array(experNum)[testExper]
        print(testDir)
        evalDir   = os.path.join(baseDir, testDir, connDir, 'eval_%s'% modelName)
        # create the directory if it doesn't already exist
        if not os.path.exists(evalDir):
            os.makedirs(evalDir)

        # initialize a dictionary with all the eval parameters for subjects
        ER = {'s%02d'%s: [] for s in sn}

        for s in sn:
            print('........ Evaluation for %s subject s%02d' % (ms, s))
            # load the model file
            models  = os.path.join(modelDir, '%s_s%02d.dat'%(modelName, s))
            MODEL   = pickle.load(open(models, "rb"))

            # Evaluate!
            [Y_roi, Ytest, Rr] = evaluate_model(MODEL, subset = [], splitby = [], 
                                                rois = {'cortex':'tesselsWB162', 'cerebellum':'grey_nan'},
                                                inclInst = inclInst, meanSubt = meanSubt, 
                                                experNum = experNum, glm = glm, avg = avg, trainMode = trainMode)
            # store all the evaluations
            ER['s%02d'%s] = Rr

            # save the evaluation 
            outname = os.path.join(evalDir, 'eval_%s_s%02d.dat'%(modelName, s))
            pickle.dump(Rr, open(outname, "wb")) # "wb": Writing Binary file
        
    return ER
    