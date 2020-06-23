#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:11:40 2020
Connectivity modelling module for connectivity project

@author: ladan
"""

# import packages
#import os
#import pandas as pd
import numpy as np
#import scipy as sp
#import data_integration as di

# importing sklearn tools. Might not need them in future!
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
#from sklearn.preprocessing import scale # ?!


# define funcitons
# OLS regression
def olsregress(X, Y):
    """
    olsregress is used to implement ols regression.
    INPUTS
    X     : The design matrix in the regression
    Y     : The response variables
    
    OUTPUTS
    M     : a dictionary with two keys: the connectivity weight estimates
            the predicted responses (Ypred)
    
    """
    M = {} # the dictionary that will contain all the info for the model
    # My code: SO SLOW!
#     M = {}
#     # estimating weights using ordinary least squares
#     W = (np.linalg.inv((X.transpose()@X)))@(X.transpose()@Y)
    
#     M['W'] = W
    
    # using sklearn
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    reg_sk = LinearRegression().fit(X, Y)
    
    ## coefficient weights + Ypred
    M['W']     = reg_sk.coef_
    M['Ypred'] = reg_sk.predict(X)
    
    
    
    return M
# ridge regressions
def ridgeregress(X, Y, method = 'l2-reg', lam = 10):
    """
    ridgereg is used to implement ridge regression.
    INPUTS
    X      : The design matrix in the regression
    Y      : The response variables
    method : the method used for ridge regression
    lam    : Lambda variable for the ridge regression
    
    OUTPUTS
    M     : a dictionary with two keys: the connectivity weight estimates
            the predicted responses (Ypred). For some models it might have other keys!
    """
    # My code: SO SLOW
    M = {} # the dictionary that will contain all the info for the model
#     # n: # of conditions
#     # p: # of cortical regions
#     # m: # of cerebellar regions (voxels)
#     [n, p] = X.shape
#     [n, m] = Y.shape
    
#     # W = (X'*X + eye(p)*lam)\(X'*Y);
#     W = np.linalg.inv((X.transpose()@X + lam*np.identity(p)))@(X.transpose()@Y)
    
#     M['W'] = W
#     M['lambda'] = lam
    
    # using sklearn
    if method == 'l2-reg': # using l2-regularization
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        ridgeReg_mod = Ridge(alpha = lam)
        reg_sk       = ridgeReg_mod.fit(X, Y)
        
    elif method == 'l1-reg': # using l1-regularization
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        lasso_mod = Lasso(alpha = lam)
        reg_sk    = lasso_mod.fit(X, Y)
        
    elif method == 'elasticNet': # combines l1 and l2 regression?
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
        elastic_mod = ElasticNet(random_state=0)
        reg_sk      = elastic_mod.fit(X, Y)
        
    ## coefficient weights + Ypred
    M['W']      = reg_sk.coef_
    M['Ypred']  = reg_sk.predict(X)
    M['lambda'] = lam
    
    
    return M
# pca regression
def pcregress(X, Y, N = 10):
    """
    pcregress is used to implement principal component regression.
    INPUTS
    X      : The design matrix in the regression
    Y      : The response variables
    N      : number of PLS component
    
    OUTPUTS
    M     : a dictionary with two keys: the connectivity weight estimates
            the predicted responses (Ypred). For some models it might have other keys!
    """
    M = {} # the dictionary that will contain all the info for the model
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

    return M
# pls regression
def plsregress(X, Y, method = 'svd', N = 10, scale = True):
    """
    plsregress is used to implement PLS regression.
    INPUTS
    X      : The design matrix in the regression
    Y      : The response variables
    method : algorithm used to estimate pls regression coeffs. can be set to 'nipals'
    N      : number of PLS component
    scale  : scale X and Y before pls regression (True) or not (False)
    
    OUTPUTS
    M     : a dictionary with two keys: the connectivity weight estimates
            the predicted responses (Ypred). For some models it might also have other keys!
    """
    M = {} # the dictionary that will contain all the info for the model
    # using sklearn
    # https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.pls.PLSRegression.html
#     from sklearn.pls import PLSCanonical, PLSRegression, CCA
    # https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
#     pls2_mod = PLSRegression(n_components = N, algorithm = method)
    pls2_mod = PLSRegression(n_components = N, scale = scale)
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
    
    return M

def R2calc(X, Y, reg):
    """
    calculates R2, R and the voxel wise versions of them.
    INPUTS: 
    X     : Regressors/explanatory variables/predictors: cortical activity profiles
    Y     : Response variable: cerebellar activity profiles
    reg   : for models estimated using sklearn, this is the output of the sklearn. In future versions of 
            the code this will change, because all we need to calculate R2 values is the estimated weights matrix
            
    OUTPUTS:
    R2
    R
    R2_vox
    R_vox
    """
    
    # if the model is estimated using sklearn (if not, you just the W or a dictionary with a key = W)
    Ypred = reg.predict(X)
    
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

