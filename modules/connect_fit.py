#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:11:40 2020
Connectivity modelling module for connectivity project

@author: ladan
"""

# import packages
import os
import pandas as pd
import numpy as np
import scipy as sp
import data_integration as di

# importing sklearn tools. Might not need them in future!
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale # ?!


# define funcitons
# OLS regression
def olsregress(X, Y):
    """
    olsregress is used to implement ols regression.
    INPUTS
    X     : The design matrix in the regression
    Y     : The response variables
    
    OUTPUTS
    reg   : output of sklearn linear regression, or the output of my own code for the model!
    Ypred : predicted values
    
    """
    # My code: SO SLOW!
#     M = {}
#     # estimating weights using ordinary least squares
#     W = (np.linalg.inv((X.transpose()@X)))@(X.transpose()@Y)
    
#     M['W'] = W
    
    # using sklearn
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    reg = LinearRegression().fit(X, Y)
    
    
    return reg
# ridge regressions
def ridgereg(X, Y, method = 'l2-reg', lam = 10):
    """
    ridgereg is used to implement ridge regression.
    INPUTS
    X     : The design matrix in the regression
    Y     : The response variables
    lam   : Lambda variable for the ridge regression
    
    OUTPUTS
    reg   : output of sklearn ridge regression, or the output of my own code for the model!
    Ypred : predicted values
    """
    # My code: SO SLOW
#     M = {}
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
        reg          = ridgeReg_mod.fit(X, Y)
        
    elif method == 'l1-reg': # using l1-regularization
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        lasso_mod = Lasso(alpha = lam)
        reg       = lasso_mod.fit(X, Y)
        
    elif method == 'elasticNet': # combines l1 and l2 regression?
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
        elastic_mod = ElasticNet(random_state=0)
        reg         = elastic_mod.fit(X, Y)
    
    
    return reg
# pca regression
def pcregress(X, Y, N = 10):
    """
    pcregress is used to implement principal component regression.
    INPUTS
    X      : The design matrix in the regression
    Y      : The response variables
    N      : number of PLS component
    
    OUTPUTS
    reg   : output of sklearn PCA regression, or the output of my own code for the model!
    Ypred : predicted values
    """
    
    # using sklearn
    ## 1. apply PCA
    pca       = PCA(n_components = N)
    X_reduced = pca.fit_transform(X)
    
    ## 2. do the regression
    reg = LinearRegression().fit(X_reduced, Y)

    return reg
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
    reg   : output of sklearn PLS regression, or the output of my own code for the model!
    Ypred : predicted values
    """
    
    # using sklearn
    # https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.pls.PLSRegression.html
#     from sklearn.pls import PLSCanonical, PLSRegression, CCA
    # https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
#     pls2_mod = PLSRegression(n_components = N, algorithm = method)
    pls2_mod = PLSRegression(n_components = N, scale = scale)
    reg      = pls2_mod.fit(X, Y)
    
    return reg

