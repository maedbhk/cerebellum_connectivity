#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:44:54 2020
contains functions that are needed for the other modules. Like indicatorMatrix

@author: ladan
"""
#import pandas as pd
import numpy as np
#import scipy as sp

# define functions
def indicatorMatrix(what, c):
    """
    creates an indicator matrix
    translating indicatorMatrix.m to python
    INPUTS
    what : what type of indicator matrix
           different options: 'identity', 'identity_fix', 'identity_p', 
           'reduced', 'reduced_p', 'pairs', 'allpairs', 'allpairs_p', 
           'interaction_reduced', 
    c    : array from which you want to create the indicator matrix
    
    OUTPUTS
    Z : the indicator matrix
    """
    
    if len(c.shape) == 1:
        c = c.reshape(-1, 1) # reshape so that it is always a column vector

    [row, col] = c.shape

    cc = np.zeros((row, col), dtype = int) # making sure that cc contains integer, otherwise I'd get typeError
    for s in np.arange(0, col):
        a, cc[:, s] = np.unique(c[:, s], return_inverse = True)# Make the class-labels 1-K 
        a = a.astype(int) # astype(int) is needed because without it, a = np.delete(a, a[a==0]) will issue a warning

    K = np.max(cc) # number of classes, assuming numbering from 1...max(c)


    if what == 'identity': # Dummy coding matrix 
        Z = np.zeros((row, K+1), dtype = int)
        for i in np.arange(0, K+1):
            ind = cc == i # without ind and reshaping it, I would get IndexError: too many indices for array
            ind = ind.reshape((cc.shape[0], ))
            Z[ind , i] = 1    

    if what == 'identity_fix': # Dummy coding matrix with fixed columns 1...K
        Z = np.zeros((row, int(np.max(c))+1), dtype = int)
        for i in np.arange(0, int(np.max(c))):
            ind = c == i+1
            ind = ind.reshape((c.shape[0], ))
            Z[ind, i+1] = 1
        # the first column corresponds to 0 and contains all zeros. It can be deleted:
        #Z = np.delete(Z, 0, axis = 1) 

    if what == 'identity_p': # Dummy coding matrix except for 0: Use this to code simple main effects 
        a = np.delete(a, a[a==0]) 
        K = len(a) 
        Z = np.zeros((row,K), dtype = int)
        for i in np.arange(0, K):
            ind = c == a[i]
            ind = ind.reshape((c.shape[0], ))
            Z[ind, i] = 1
            
    if what == 'reduced': # Reduced rank dummy coding matrix (last category all negative) 
        Z = np.zeros((row, K), dtype = int);
        for  i in np.arange(0, K):
            ind = cc == i
            ind = ind.reshape((cc.shape[0], ))
            Z[ind, i] = 1
        Z[np.sum(Z, axis = 1) == 0, :] = -1
    
    if what == 'reduced_p': # Reduced rank dummy coding matrix, except for 0
        a = np.delete(a, a[a==0]) 
        K = len(a) 
        Z = np.zeros((row,K-1), dtype = int)
        for i in np.arange(0, K-1):
            ind = c == a[i]
            ind = ind.reshape((c.shape[0], ))
            Z[ind, i] = 1
        ind2 = c == a[K-1]
        ind2 = ind2.reshape((c.shape[0], ))
        Z[ind2, :] = -1
        
    if what == 'pairs': # K-1 pairs 
        Z = np.zeros((row,K), dtype = float)
        for i in np.arange(0, K):
            ind1 = cc == i
            ind2 = cc == i+1
            ind1 = ind1.reshape((cc.shape[0], ))
            ind2 = ind2.reshape((cc.shape[0], ))
            Z[ind1, i] =  1/np.sum(ind1)
            Z[ind2, i] = -1/np.sum(ind2)

    if what =='allpairs': # all possible pairs
        Knew = K+1
        Z = np.zeros((row,int(Knew*(Knew-1)/2)), dtype = float)
        k = 0
        for i in np.arange(0, K+1):
            for j in np.arange(i+1, K+1):
                indi = cc == i
                indj = cc == j
                indi = indi.reshape((cc.shape[0], ))
                indj = indj.reshape((cc.shape[0], )) 
                Z[indi, k] = 1/sum(indi)
                Z[indj, k] = -1/sum(indj)
                k          = k + 1
                
    if what == 'allpairs_p': # all possible pairs  except for 0 
        a = np.delete(a, a[a==0])  
        K = len(a) 
        Z = np.zeros((row,int(K*(K-1)/2)), dtype = float)
        k = 0
        for i in np.arange(0, K+1):
            for j in np.arange(i+1, K):
                indai = c == a[i]
                indaj = c == a[j]
                indai = indai.reshape((c.shape[0], ))
                indaj = indaj.reshape((c.shape[0], ))
                Z[indai,k] = 1/np.sum(indai)
                Z[indaj,k] = -1/np.sum(indaj)
                k          = k+1

    if what == 'interaction_reduced': #### not checked
        Z1 = indicatorMatrix('reduced',cc[:,0])
        Z2 = indicatorMatrix('reduced',cc[:,1]) 
        for n in np.arange(0, row):
            Z[n,:] = np.kron[Z1[n,:],Z2[n,:]]

    if what == 'hierarchical': # Allows for a random effects f c(:,2) within each level of c(:,1)
        Z1 = indicatorMatrix('identity', cc[:,0]) 
        Z2 = indicatorMatrix('reduced', cc[:,1])
        for n in np.arange(0, row):
            Z[n,:] = np.kron(Z1[n,:],Z2[n,:]) 

    if what == 'hierarchicalI': # Allows for a random effects f c(:,2) within each level of c(:,1)
        C = cc.shape[1] 
        Z = {} # instead of cell array in matlab, I'm using a dictionary
        for i in np.arange(0, C):
            Z[i] = indicatorMatrix('identity',cc[:,i])
        A = Z[list(Z.keys())[-1]]
        for i in np.arange(C-1, -1, 1):
            B = []; # not the right size
            for n in np.arange(0, row):
                B[n,:] = np.kron(Z[i][n,:],A[n,:])
            A = B
        Z = A
    
    return Z

