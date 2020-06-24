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

# import sklearn
## once I have my own functions I will get rid of this section
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
# from sklearn.preprocessing import scale # ?!

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
def connect_fit(X, Y, model, **kwargs):
    """
    connect_fit fits different models and returns the parameters of the model alongside the predicted values
    INPUTS
    X     : The design matrix in the regression
    Y     : The response variables
    model : the connectivity model. Options are:
            'olsregress' no other parameters
            'l2regress'  lam
            'l1regress'  lam
            'elasticnet' lam
            'pcregress'  N
            'plsregress' N
            'regbicluster'
    
    OUTPUTS
    M     : a dictionary with two keys: the connectivity weight estimates
            the predicted responses (Ypred). 
            The keys in M will depend on the model
    
    """
    if model == 'olsregress':     # ols regression
        M = {} # the dictionary that will contain all the info for the model
        # My code: SO SLOW!
#         # estimating weights using ordinary least squares
#         W      = (np.linalg.inv((X.transpose()@X)))@(X.transpose()@Y)
#         M['W'] = W

        # using sklearn
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        reg_sk = LinearRegression().fit(X, Y)

        ## coefficient weights + Ypred
        M['W']     = reg_sk.coef_
        M['Ypred'] = reg_sk.predict(X)
        
    elif model == 'l2regress':    # l2 ridge regression
        M   = {}
        lam = args
        # using sklearn
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        ridgeReg_mod = Ridge(alpha = lam)
        reg_sk       = ridgeReg_mod.fit(X, Y)
        
        ## coefficient weights + Ypred
        M['W']      = reg_sk.coef_
        M['Ypred']  = reg_sk.predict(X)
        M['lambda'] = lam
        
    elif model == 'l1regress':    # l1 ridge regression
        M   = {}
        lam = args
        # using sklearn
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        lasso_mod = Lasso(alpha = lam)
        reg_sk    = lasso_mod.fit(X, Y)
        
        ## coefficient weights + Ypred
        M['W']      = reg_sk.coef_
        M['Ypred']  = reg_sk.predict(X)
        M['lambda'] = lam
        
    elif model == 'elasticnet':   # elastic net ridge regression
        M   = {}
        lam = args
        # using sklearn
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
        elastic_mod = ElasticNet(random_state=0)
        reg_sk      = elastic_mod.fit(X, Y)
        
        ## coefficient weights + Ypred
        M['W']      = reg_sk.coef_
        M['Ypred']  = reg_sk.predict(X)
        M['lambda'] = lam
        
    elif model == 'pcregress':    # principal component regression
        M = {} # the dictionary that will contain all the info for the model
        N = args
        # using sklearn
        ## 1. apply PCA
        pca       = PCA(n_components = N)
        X_reduced = pca.fit_transform(X)

        ## 2. do the regression
        reg_sk = LinearRegression().fit(X_reduced, Y)

        ## coefficient weights + Ypred
        M['W']     = reg_sk.coef_
        M['Ypred'] = reg_sk.predict(X)
        M['nPC']   = N
        
    elif model == 'plsregress':   # pls regression
        M = {} # the dictionary that will contain all the info for the model
        N = args
        # using sklearn
        # https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.pls.PLSRegression.html
#         from sklearn.pls import PLSCanonical, PLSRegression, CCA
        # https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
#         pls2_mod = PLSRegression(n_components = N, algorithm = method)
        pls2_mod = PLSRegression(n_components = N, scale = True)
        reg_sk   = pls2_mod.fit(X, Y)

        ## coefficient weights + Ypred
        M['W']     = reg_sk.coef_
        M['Ypred'] = reg_sk.predict(X)
        M['nPLS']  = N
        M['XL']    = reg_sk.x_loadings_ # X loadings
        M['XS']    = reg_sk.x_scores_   # X scores
        M['XW']    = reg_sk.x_weights_  # weights used to project X to the latent structure
        M['YL']    = reg_sk.y_loadings_ # Y loadings
        M['YS']    = reg_sk.y_scores_   # Y scores
        M['YW']    = reg_sk.y_weights_  # weights used to project Y to the latent structure
        
    elif model == 'regbicluster': # regression and biclustering??????????????????????!!!!!!!!!!!!!!!!!!!!
        print('STILL UNDER CONSTRUCTION!!!!')
        
    return M

def R2calc(X, Y, M):
    """
    calculates R2, R and the voxel wise versions of them.
    INPUTS: 
    X     : Regressors/explanatory variables/predictors: cortical activity profiles
    Y     : Response variable: cerebellar activity profiles
    M     : The dictionary returned by the modelling function containing connectivity weights and Ypred
            
    OUTPUTS:
    R2
    R
    R2_vox
    R_vox
    """
    
    # if the model is estimated using sklearn (if not, you just the W or a dictionary with a key = W)
    Ypred = M['Ypred']
    
    # Calculating R2
    res = Y - Ypred
    SSR = np.nansum(res **2, axis = 0) # remember: without setting the axis, it just "flats" out the whole array and sum over all
    SST = np.nansum(Y*Y, axis = 0)
    
    R2_vox = 1 - (SSR/SST)
    R2     = 1 - (np.nansum(SSR)/np.nansum(SST))
    
    # Calculating R 
    SYP = np.nansum(Y*Ypred, axis = 0);
    SPP = np.nansum(Ypred*Ypred, axis = 0);

    R     = np.nansum(SYP)/np.sqrt(np.nansum(SST)*np.nansum(SPP));
    R_vox = SYP/np.sqrt(SST*SPP) # per voxel

    return (R2, R, R2_vox, R_vox)
    

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