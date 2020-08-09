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
import pandas as pd
import numpy as np
#import scipy as sp
from sklearn.preprocessing import StandardScaler # for scaling the X
import prep_data # module used to prepare data
# import packages for visualizations
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt 
# %matplotlib inline
# import seaborn as sns

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
## takes model info as input and returns evaluation paprameters
def evaluate_model(Md, subset = [], splitby = [], rois = {'cortex':'tesselsWB162', 'cerebellum':'grey_nan'},
             inclInst = 1, meanSubt = 1, experNum = [1, 2], glm = 7, avg = 1, trainMode = 'crossed'):
    
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
    - trainMode : doubled crossed cross validation crossed ('crossed') or uncrossed ('uncrossed')
    
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
          'Rcvvox':[], 'Rncvox':[], 'Ryvox':[], 'Rpvox':[],
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
                
                SSP     = sum(sum(predY[:,goodindx]**2))                   # Sum of square of predictions
                SSPvox  = sum(predY**2)
                SSY     = sum(sum(Y_roi[testBindx,:]**2))                  # Sum of squares of data
                SSYvox  = sum(Y_roi[testBindx,:]**2)
                SSCp    = sum(sum(predY[:,goodindx]*predYnc[:,goodindx]))  # Covariance of Predictions
                SSCpvox = sum(predY*predYnc)
                SSCy    = sum(sum(Y_roi_good*Y_roi_good))                  # Covariance of Y's
                SSCyvox = sum(Y_roi[testBindx,:]*Y_roi[testBindx,:])
                SSCn    = sum(sum(predYnc[:,goodindx]*Y_roi_good))         # Covariance of non-cross prediction and data
                SSCnvox = sum(predYnc*Y_roi[testBindx,:])
                SSCc    = sum(sum(predY[:,goodindx]*Y_roi_good))           # Covariance of cross prediction and data
                SSCcvox = sum(predY*Y_roi[testBindx,:])
                
                RR['sn'].append(Md['sn'][im])
                RR['params'].append(Md['params'][im])
                RR['model'].append(Md['model'][im])
                RR['trainMode'].append(Md['trainMode'][im])
                RR['xname'].append(rois['cortex'])
                
                RR['Rcv'].append(SSCc / np.sqrt(SSY*SSP)) # Double-Crossvalidated predictive correlation
                print(f"Rcv is {RR['Rcv']}")
                RR['Rcvvox'].append(SSCcvox/np.sqrt(SSYvox*SSPvox))
                RR['Rnc'].append(SSCn / np.sqrt(SSY*SSP)) # Not double-crossvalidated predictive correlation
                RR['Rncvox'].append(SSCnvox/np.sqrt(SSYvox*SSPvox))
                print(f"Rnc is {RR['Rnc']}")
                
                # If we knew the true pattern, the best correlation we
                # would expect is sqrt(Ry) 
                # We also want to take into account the relaibility 
                RR['Ry'].append(SSCy / SSY)               # Reliability of data
                print(f"Ry is {RR['Ry']}")
                RR['Ryvox'].append(SSCyvox/SSYvox)
                RR['Rp'].append(SSCp / SSP)               # Reliability of prediction
                print(f"Rp is {RR['Rp']}")
                RR['Rpvox'].append(SSCpvox/SSPvox)
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


    Ytest = Y_roi[testBindx,:]
                    
    
    return (Ytest, predY, RR)

# returns and saves evaluation parameters for all the subjects in the input
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
    - Y        : a dictionary containing Ytest and Ypred which can be used later for RDMs?
    
    """
    
    # setting directories
    for ms in model: # ms: model string
        
        modelName = 'mb4_%s_%s'% (rois['cortex'], ms)
        modelDir  = os.path.join(baseDir, 'sc%d'% trainExper, connDir, 'glm%d'%glm, modelName)

        # get the testExper
        testExper = [item != trainExper for item in experNum] # get the test Experiment
        testDir   = 'sc%d'% np.array(experNum)[testExper]
        evalDir   = os.path.join(baseDir, testDir, connDir, 'glm%d'%glm, 'eval_%s'% modelName)
        YtestDir  = os.path.join(baseDir, testDir, connDir, 'glm%d'%glm, 'Y_%s'% modelName)
        
        # create the directory if it doesn't already exist
        if not os.path.exists(evalDir):
            os.makedirs(evalDir)
        if not os.path.exists(YtestDir):
            os.makedirs(YtestDir)
            

        # initialize a dictionary with all the eval parameters for subjects
        ER = {'s%02d'%s: [] for s in sn}
        Y  = {'s%02d'%s: [] for s in sn}

        for s in sn:
            
            Y['s%02d'%s] = {'Ytest':[], 'Ypred':[]} # saving Ytest and Ypred in a dictionary  
            print('........ Evaluation for %s subject s%02d' % (ms, s))
            # load the model file
            models  = os.path.join(modelDir, '%s_s%02d.dat'%(modelName, s))
            MODEL   = pickle.load(open(models, "rb"))

            # Evaluate!
            [Ytest, Ypred, Rr] = evaluate_model(MODEL, subset = [], splitby = [], 
                                                rois = rois,
                                                inclInst = inclInst, meanSubt = meanSubt, 
                                                experNum = experNum, glm = glm, avg = avg, trainMode = trainMode)
            # store all the evaluations
            ER['s%02d'%s]         = Rr
            Y['s%02d'%s]['Ytest'] = Ytest
            Y['s%02d'%s]['Ypred'] = Ypred

            # save the evaluation 
            outname_ER = os.path.join(evalDir, 'eval_%s_s%02d.dat'%(modelName, s))
            outname_Y  = os.path.join(YtestDir, 'Y_%s_s%02d.dat'%(modelName, s))
            pickle.dump(Rr, open(outname_ER, "wb")) # "wb": Writing Binary file
            pickle.dump(Y, open(outname_Y, "wb")) # "wb": Writing Binary file
        
    return ER, Y

## calculates RDMs using original and predicted test datasets
def rdm_calc(Y, model, rois = {'cortex':'tesselsWB162', 'cerebellum':'grey_nan'}):
    """
    Calculates RDMs using predicted data + Calculates RDMs using actual data
    INPUTS
    Y     : Y dictionary containing Ytest and Ypred 
    model : the model for which you want to calculate RDMs
    rois  : roi dictionary containing rois used in modelling
    
    OUTPUTS
    RDM  : RDM calculated using actual data
    pRDM : RDM calculated using predicted data
    
    """
    
    return (RDM, pRDM)

## creates and returns a dataframe which can be used in plotting ...
def eval_df(sn, glm = 7, models = ['l2regress', 'plsregress'], 
                 rois = ['tesselsWB162', 'tesselsWB362', 'tesselsWB642'], 
                 testExper = 2):
    """
    This function plot a toy plot: measure vs subject.
    IINPUTS
    sn        : subjects on the x-axis
    glm       : glmNumber
    models    : a list containing models I want to be included in the plot
    rois      : a list containing cortical rois used 
    testExper : Experiment used for testing
    
    OUTPUTS
    df        : dataframe with all the info for the models and cortical ROIs
    
    
    """
    # create a dataframe with all the data 
    tmpF = []
    for rs in rois:
        for ms in models:
            # the directory where the model data is saved
            evalName = 'eval_mb4_%s_%s'% (rs, ms)
            evalDir   = os.path.join(baseDir, 'sc%d'%testExper, connDir, 'glm%d'%glm, evalName)
            
            for s in sn:
                # load the dictionary for each subject and put it in a dataframe
                # dataframes will be concatenated
                evalNames = os.path.join(evalDir, '%s_s%02d.dat'%(evalName, s))
                EV        = pickle.load(open(evalNames, "rb"))
                
                # Discard the voxel-wise measures
                EV.pop('Rcvvox')
                EV.pop('Rncvox')
                EV.pop('Ryvox')
                EV.pop('Rpvox')

                ef        = pd.DataFrame(EV)
                tmpF.append(ef)
    df = pd.concat(tmpF, ignore_index=True)
    
    # this line here handles the issue with the legend when using seaborn.lineplot.
    # more on the issue here: https://github.com/mwaskom/seaborn/issues/1653
#     df["params"] = ["$%s$" % x for x in df["params"]]

    return df
    