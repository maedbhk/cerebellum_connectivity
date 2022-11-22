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
    Xdata = Dataset(experiment = "sc1", glm = "glm7", roi = cortex, subj_id = "all") # Any list of subjects will do (experiment=experiment, roi='cerebellum_suit', subj_id=s)
    Xdata.load_h5()  
    X, X_info = Xdata.get_data()                           # Load from Matlab
    alpha_ridge = np.exp(logalpha)

    # calculate VIF 
    term1 = (X.T@X + alpha_ridge*np.eye(X.shape[1]))
    term2 = X.T@X
    var_B = np.linalg.inv(term1)@ term2 @ np.linalg.inv(term1)
    vif  = np.diag(var_B)

    # create and save the cortical map
    func_giis, hem_names = cdata.convert_cortex_to_gifti(
                                                            vif, 
                                                            atlas = cortex,
                                                            data_type='func',
                                                            column_names=None,
                                                            label_names=None,
                                                            label_RGBA=None,
                                                            hem_names=['L', 'R'])

    for (func_gii, hem) in zip(func_giis, hem_names):
        nib.save(func_gii, os.path.join(const.base_dir,f'vif_{cortex}_logalpha{logalpha}.{hem}.func.gii'))
    return


if __name__ == "__main__":
    calc_vif(cortex = "tessels1002", logalpha = 8, sn = const.return_subjs)