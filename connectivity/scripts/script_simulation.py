import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import os
import connectivity.constants as const
from connectivity.data import Dataset
import connectivity.model as model
import connectivity.data as data
import connectivity.run as run
import connectivity.visualize as vis
import connectivity.figures as fig
import connectivity.io as cio
import connectivity.evaluation as eval
from SUITPy import flatmap
import itertools
import nibabel as nib
import h5py
import deepdish as dd
import seaborn as sns

def sim_from_cortex(atlas='tessels0042',sub = 's02',sigma=0.5):
    """Generates an artificial data set assuming the one-to-one connectivity model for each cortical parcel. 
    """
    Xdata = Dataset('sc1','glm7',atlas,sub)
    Xdata.load_mat() # Load from Matlab 
    X, INFO = Xdata.get_data(averaging="sess") # Get numpy 
    # Get the test data set 
    XTdata = Dataset('sc2','glm7',atlas,sub)
    XTdata.load_mat() # Load from Matlab 
    XT, INFO = Xdata.get_data(averaging="sess") # Get numpy 

    i1 = np.where(INFO.sess==1)
    i2 = np.where(INFO.sess==2)
    rel = np.sum(X[i1,:]*X[i2,:])/np.sqrt(np.sum(X[i1,:]**2) * np.sum(X[i2,:]**2))
    D=pd.DataFrame()
    N,Q = X.shape
    P = 1000
    MOD =[]
    MOD.append(model.L2regression(alpha=1))
    MOD.append(model.Lasso(alpha=0.1))
    MOD.append(model.WTA())
    model_name = ['ridge','lasso','WTA']

    for i in range(Q): 
        Y = X[:,i].reshape(N,1) + np.random.normal(0,sigma,(N,P))
        Y1 = X[:,i].reshape(N,1) + np.random.normal(0,sigma,(N,P)) # Within sample replication
        Y2 = XT[:,i].reshape(N,1) + np.random.normal(0,sigma,(N,P)) # Out of sample 
        for m in range(len(MOD)):
            MOD[m].fit(X,Y)
            Ypred1 = MOD[m].predict(X)
            Ypred2 = MOD[m].predict(XT)
            r1,_ = eval.calculate_R(Y1,Ypred1)
            r2,_ = eval.calculate_R(Y2,Ypred2)
            T=pd.DataFrame({'atlas':atlas,
                            'sub':sub,
                            'Node':[i],
                            'model':model_name[m],
                            'modelNum':[m],
                            'Rin':[r1],
                            'Rout':[r2]})
            D=pd.concat([D,T])
    return D

if __name__ == "__main__":
    D= sim_from_cortex()
    plt.subplot(1,2,1)
    sns.barplot(data=D,x='model',y='Rin')
    plt.subplot(1,2,2)
    sns.barplot(data=D,x='model',y='Rout')

    pass
