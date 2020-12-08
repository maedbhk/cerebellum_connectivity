import os
import numpy as np


"""
Created on Aug 10 10:04:24 2020
Evaluation of connectivity models

@authors: Maedbh King, Ladan Shahshahani, and Joern diedrichsen 
"""

def calculate_R(Y, Y_pred):
    # Calculating R 
    res = Y - Y_pred

    SYP = np.nansum(Y*Y_pred, axis = 0);
    SPP = np.nansum(Y_pred*Y_pred, axis = 0);
    SST = np.sum((Y**2, axis = 0) # use np.nanmean(Y) here?

    R = np.nansum(SYP)/np.sqrt(np.nansum(SST)*np.nansum(SPP));
    R_vox = SYP/np.sqrt(SST*SPP) # per voxel

    return R, R_vox

def calculate_R2(Y, Y_pred):
    res = Y - Y_pred

    SSR = np.nansum(res **2, axis = 0) # remember: without setting the axis, it just "flats" out the whole array and sum over all
    SST = np.sum(Y** 2, axis = 0) # use np.nanmean(Y) here??

    R2 = 1 - (np.nansum(SSR)/np.nansum(SST))
    R2_vox = 1 - (SSR/SST)

    return R2, R2_vox

def calculate_noiseceiling(Y, T):
    Yflip=np.r_[Y(T.sess==2,:),Y(T.sess==1)]
    R, Rvox = calculate_R(Y, Yflip)
    R2, R2vox = calculate_R2(Y, Yflip)
    return R, R_vox, R2, R2_vox