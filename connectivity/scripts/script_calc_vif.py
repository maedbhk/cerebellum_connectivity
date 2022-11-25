import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import os
import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.data as cdata
import connectivity.run as run
import connectivity.visualize as vis
import connectivity.figures as fig
import connectivity.io as cio
from SUITPy import flatmap
import itertools
import nibabel as nib
import h5py
import deepdish as dd


def calc_vif(cortex = "tessels1002", logalpha = 8, sn = const.return_subjs):
    # (X′X+λIp)^(−1)X′X(X′X+λIp)^(−1)
    # get cortical data for the average subject
    vif = []
    for i,s in enumerate(sn):
        print(f'calculcating {s}')
        Xdata = Dataset(experiment = "sc1", glm = "glm7", 
                        roi = cortex, 
                        subj_id = s) 
        Xdata.load_mat()
        X, X_info = Xdata.get_data()                           # Load from Matlab
        # In the normal Ridge model we standarize the data, so we need to do this here as well 
        scale = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
        X = X / scale

        alpha_ridge = np.exp(logalpha)

        # calculate Variance under the full model and under the diagonal model
        good = np.logical_not(np.isnan(X.sum(axis=0)))
        X = X[:,good]
        XX = X.T@X  # Gram Matrix 
        dXX = np.diag(np.diag(XX)) # diagnonal of the gram matrix 
        Pa = (XX + alpha_ridge*np.eye(X.shape[1]))
        Pb = (dXX + alpha_ridge*np.eye(X.shape[1]))
        var_Ba = np.linalg.inv(Pa)@ XX @ np.linalg.inv(Pa)
        var_Bb = np.linalg.inv(Pb)@ XX @ np.linalg.inv(Pb)
        v = np.diag(var_Ba)/np.diag(var_Bb)
        if i==0:
            vif = np.zeros((len(good),len(sn)))*np.nan
        vif[good,i]=v
    m = np.nanmean(vif,axis=1,keepdims=True)
    vif = np.concatenate([vif,m],axis=1)

    # create and save the cortical map
    func_giis, hem_names = cdata.convert_cortex_to_gifti(
                                vif, 
                                atlas = cortex,
                                data_type='func',
                                column_names=sn+['mean'],
                                label_names=None,
                                label_RGBA=None,
                                hem_names=['L', 'R'])

    for (func_gii, hem) in zip(func_giis, hem_names):
        nib.save(func_gii, os.path.join(const.base_dir,f'sc1/conn_models/vif_{cortex}_logalpha{logalpha}.{hem}.func.gii'))
    return

def calc_snr(cortex = "tessels1002", sn = const.return_subjs):

    # (X′X+λIp)^(−1)X′X(X′X+λIp)^(−1)
    # get cortical data for the average subject
    Xdata = Dataset(experiment = "sc1", glm = "glm7", roi = cortex, subj_id = "all") # Any list of subjects will do (experiment=experiment, roi='cerebellum_suit', subj_id=s)
    Xdata.load_h5()
    X, X_info = Xdata.get_data()                           
    snr = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
    # create and save the cortical map
    func_giis, hem_names = cdata.convert_cortex_to_gifti(
                                                            snr, 
                                                            atlas = cortex,
                                                            data_type='func',
                                                            column_names=None,
                                                            label_names=None,
                                                            label_RGBA=None,
                                                            hem_names=['L', 'R'])

    for (func_gii, hem) in zip(func_giis, hem_names):
        nib.save(func_gii, os.path.join(const.base_dir,f'sc1/conn_models/snr_{cortex}.{hem}.func.gii'))
    return

def calc_vif_lambda(cortex = "tessels0162", 
            logalpha = np.linspace(-5,10,15)):

    # (X′X+λIp)^(−1)X′X(X′X+λIp)^(−1)
    # get cortical data for the average subject
    Xdata = Dataset(experiment = "sc1", glm = "glm7", roi = cortex, subj_id = "all") # Any list of subjects will do (experiment=experiment, roi='cerebellum_suit', subj_id=s)
    Xdata.load_h5()
    X, X_info = Xdata.get_data()                           # Load from Matlab
    # In the normal Ridge model we standarize the data, so we need to do this here as well 
    scale = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
    X = X / scale

    meanVIF = np.zeros(logalpha.shape)
    minVIF = np.zeros(logalpha.shape)
    maxVIF = np.zeros(logalpha.shape)
    stdVIF = np.zeros(logalpha.shape)

    for i,la in enumerate(logalpha):
        alpha_ridge = np.exp(la)

        # calculate Variance under the full model and under the diagonal model
        XX = X.T@X  # Gram Matrix 
        dXX = np.diag(np.diag(XX)) # diagnonal of the gram matrix 
        Pa = (XX + alpha_ridge*np.eye(X.shape[1]))
        Pb = (dXX + alpha_ridge*np.eye(X.shape[1]))
        var_Ba = np.linalg.inv(Pa)@ XX @ np.linalg.inv(Pa)
        var_Bb = np.linalg.inv(Pb)@ XX @ np.linalg.inv(Pb)
        vif  = np.diag(var_Ba)/np.diag(var_Bb)
        meanVIF[i] = np.nanmean(vif)
        maxVIF[i] = vif.max()
        minVIF[i] = vif.min()
        stdVIF[i] = vif.std()

    pass
    plt.plot(logalpha,meanVIF,'k')
    plt.plot(logalpha,minVIF,'k:')
    plt.plot(logalpha,maxVIF,'k:')

    return


if __name__ == "__main__":
    # calc_vif_lambda(cortex = "tessels0642")
    calc_vif(cortex = "tessels1002", logalpha = 8, sn = const.return_subjs)
    # calc_snr(cortex = "tessels1002")