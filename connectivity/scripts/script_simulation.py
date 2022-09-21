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
import connectivity.matrix as matrix
from SUITPy import flatmap
import itertools
import nibabel as nib
import nilearn.plotting as nip
import h5py
import deepdish as dd
import seaborn as sns
from sklearn.model_selection import cross_val_score
import connectivity.nib_utils as nio
import PcmPy as pcm 
from numpy import kron,ones,zeros,random, sum, sqrt

def getX_random(N=60,Q=80):
    """Generates an artificial data set using iid data for Cortex
    """
    X1 = np.random.normal(0,1,(N,Q))
    X2 = np.random.normal(0,1,(N,Q))
    return X1, X2

def getX_clusters(N=20,Q=40,K=10,eps=0.4):
    """ Make 2 clustered data sets with random
    Args:
        N (int): Number of observations
        Q (int): Number of cortical voxels
        K (int): Number of cluster
    """
    k = kron(np.arange(K),ones(int(Q/K),))
    U = matrix.indicator(k).T
    V = random.normal(0,1,size=(N,K))
    V = V/sqrt(sum(V**2,axis=0))
    s = random.chisquare(df=4,size=(Q,))
    X1 = (V@U)*s+random.normal(0,1,size=(N,Q))*eps
    X2 = (V@U)*s+random.normal(0,1,size=(N,Q))*eps
    return X1,X2

def getX_cortex(atlas='tessels0042',sub = 's02'):
    """Generates an artificial data set using real cortical data
    """
    Xdata = Dataset('sc1','glm7',atlas,sub)
    Xdata.load_mat() # Load from Matlab
    X1, INFO1 = Xdata.get_data(averaging="sess") # Get numpy
    # Get the test data set
    XTdata = Dataset('sc2','glm7',atlas,sub)
    XTdata.load_mat() # Load from Matlab
    X2, INFO2 = XTdata.get_data(averaging="sess") # Get numpy
    # z-standardize cortical regressors
    X1 = X1 / np.sqrt(np.sum(X1 ** 2, 0) / X1.shape[0])
    X2 = X2 / np.sqrt(np.sum(X2 ** 2, 0) / X1.shape[0])
    X1 = np.nan_to_num(X1)
    X2 = np.nan_to_num(X2)
    # i1 = np.where(INFO1.sess==1)
    # i2 = np.where(INFO1.sess==2)
    # rel = np.sum(X1[i1,:]*X1[i2,:])/np.sqrt(np.sum(X1[i1,:]**2) * np.sum(X1[i2,:]**2))
    return X1,X2,INFO1,INFO2

def getW(P,Q,conn_type='one2one',X=None,sparse_num=2):
    """_summary_

    Args:
        P (int): number of cerebellar voxels
        Q (int): number of cortical oarceks
        conn_type (str): Connectivity type
            - 'one2one': One-to-one connectivity
            - 'sparse': select sparse_num random parcels for each voxel
            - 'laplace': Laplace distribution (L1-equivalent)
            - 'normal': Normal distribution (L2-equivalent)
        X (nd-array): Designmmartrix when given, ensures that predicted profiles are unit length
        sparse_prob: (float): Defaults to 0.05. For sparse only

    Returns:
        W: (nd.array)
    """
    if conn_type=='one2one':
        k = np.int(np.ceil(P/Q))
        W = np.kron(np.ones((k,1)),np.eye(Q))
        W = W[0:P,0:Q]
    elif conn_type=='sparse':
        num=np.max([1,sparse_num])
        W=np.zeros((P,Q))
        for i in range(W.shape[0]):
            ind = np.random.choice(Q,size=(num,1),replace=True)
            W[i,ind]=1
    elif conn_type=='laplace':
        W = np.random.laplace(0,1,size=(P,Q))
    elif conn_type=='normal':
        W=np.random.normal(0,0.2,(P,Q))
    if X is not None:
        p = X @ W.T
        w = np.sqrt(np.sum(p**2,axis=0))
        print(f"zeros={np.sum(w==0)/w.shape[0]:.3f}")
        w[w==0]=1
        W = W / w.reshape((-1,1))
    return W


def gridsearch(modelclass,log_alpha,X,Y):
    r_cv = np.empty((len(log_alpha),))
    for i,a in enumerate(log_alpha):
        model = modelclass(alpha=np.exp(a))
        a = cross_val_score(model, X, Y, scoring=eval.calculate_R_cv, cv=4)
        r_cv[i] = a.mean()
    indx = r_cv.argmax()
    return log_alpha[indx],r_cv

def sim_random(N=60,Q=80,P=1000,sigma=0.1,conn_type='one2one'):
    #  alphaR = validate_hyper(X,Y,model.L2regression)
    D=pd.DataFrame()
    X1,X2 = getX_random(N,Q)
    W = getW(P,Q,conn_type,X=X1)
    Y1  = X1 @ W.T + np.random.normal(0,sigma,(N,P))
    Y1a = X1 @ W.T + np.random.normal(0,sigma,(N,P)) # Within sample replication
    Y2 = X2 @ W.T  + np.random.normal(0,sigma,(N,P)) # Out of sample

    # Tune hyper parameters for Ridge and Lasso model
    logalpha_ridge, r_cv_r = gridsearch(model.L2regression,[-2,0,2,4,6,7,8,9,10,11,12],X1,Y1)
    logalpha_lasso, r_cv_l = gridsearch(model.Lasso,[-7,-6.5,-6,-5.5,-5,-4.5,-4,-3,-2],X1,Y1)

    MOD =[]
    MOD.append(model.L2regression(alpha=np.exp(logalpha_ridge)))
    MOD.append(model.Lasso(alpha=np.exp(logalpha_lasso)))
    MOD.append(model.WTA())
    model_name = ['ridge','lasso','WTA']
    logalpha  = [logalpha_ridge,logalpha_lasso,np.nan]

    for m in range(len(MOD)):

        MOD[m].fit(X1,Y1)
        Ypred1 = MOD[m].predict(X1)
        Ypred2 = MOD[m].predict(X2)
        r1,_ = eval.calculate_R(Y1a,Ypred1)
        r2,_ = eval.calculate_R(Y2,Ypred2)
        T=pd.DataFrame({'conn_type':[conn_type],
                    'model':[model_name[m]],
                    'modelNum':[m],
                    'numtessels':[Q],
                    'logalpha':[logalpha[m]],
                    'Rin':[r1],
                    'Rout':[r2]})
        D=pd.concat([D,T])
    return D


def sim_cortical(P=2000,atlas='tessels0042',sub = 's02',
                sigma=0.1,conn_type='one2one'):
    #  alphaR = validate_hyper(X,Y,model.L2regression)
    D=pd.DataFrame()
    X1,X2,I1,I2 = getX_cortex(atlas,sub)
    N1,Q = X1.shape
    N2,_ = X2.shape
    W = getW(P,Q,conn_type,X=X1)
    Y1  = X1 @ W.T + np.random.normal(0,sigma,(N1,P))
    Y1a = X1 @ W.T + np.random.normal(0,sigma,(N1,P)) # Within sample replication
    Y2 = X2 @ W.T  + np.random.normal(0,sigma,(N2,P)) # Out of sample

    # Tune hyper parameters for Ridge and Lasso model
    logalpha_ridge, r_cv_r = gridsearch(model.L2regression,[0,2,4,6,8,10,12],X1,Y1)
    logalpha_lasso, r_cv_l = gridsearch(model.Lasso,[-5,-4,-3,-2,-1,0,1],X1,Y1)

    MOD =[]
    MOD.append(model.L2regression(alpha=np.exp(logalpha_ridge)))
    MOD.append(model.Lasso(alpha=np.exp(logalpha_lasso)))
    MOD.append(model.WTA())
    model_name = ['ridge','lasso','WTA']
    logalpha  = [logalpha_ridge,logalpha_lasso,np.nan]

    for m in range(len(MOD)):

        MOD[m].fit(X1,Y1)
        Ypred1 = MOD[m].predict(X1)
        Ypred2 = MOD[m].predict(X2)
        r1,_ = eval.calculate_R(Y1a,Ypred1)
        r2,_ = eval.calculate_R(Y2,Ypred2)
        T=pd.DataFrame({'conn_type':[conn_type],
                    'model':[model_name[m]],
                    'modelNum':[m],
                    'numtessels':[Q],
                    'atlas':[atlas],
                    'sub':[sub],
                    'logalpha':[logalpha[m]],
                    'Rin':[r1],
                    'Rout':[r2]})
        D=pd.concat([D,T])
    return D

def sim_scenario1():
    conn_type=['laplace','sparse','one2one','normal']
    sigma = 0.3
    D=pd.DataFrame()
    Q =[7,40,80,160,240] # ,160,240
    for i,ct in enumerate(conn_type):
        for q in Q:
            print(f"{ct} for {q}")
            T = sim_random(Q=q,sigma=sigma,conn_type=ct)
            D=pd.concat([D,T],ignore_index=True)
    D.to_csv('notebooks/simulation_iid.csv')
    return D

def sim_scenario2():
    conn_type=['one2one','sparse','normal']
    atlas = ['tessels0642','tessels1002']
    sigma = 0.25
    sn = const.return_subjs
    for a in atlas:
        D=pd.DataFrame()
        for i,ct in enumerate(conn_type):
            for s in sn:
                print(f"{ct} for {a} for {s}")
                T = sim_cortical(sigma=sigma,conn_type=ct,atlas=a,sub=s)
                D=pd.concat([D,T],ignore_index=True)
        D.to_csv('notebooks/simulation_cortex_' + a + '.csv')
    return D

def plot_sim_scenario2(): 
    atlas = ['tessels0042','tessels0162','tessels0362','tessels0642','tessels1002']
    conn_type=['one2one','normal']
    T=pd.DataFrame()
    for a in atlas: 
        D = pd.read_csv(f'notebooks/simulation_cortex_{a}.csv')
        T=pd.concat([T,D])
    
    # conn_type = np.unique(D.conn_type)
    plt.style.use('seaborn-poster') 
    params = {'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'font.weight': 'regular',
            'font.family': 'sans-serif',
            'lines.markersize': 3,
            'lines.linewidth': 3
            }
    plt.rcParams.update(params)
    plt.figure(figsize=(7,3.4))

    for i,ct in enumerate(conn_type):
        plt.subplot(1,2,i+1)
        # plt.style.use('seaborn-poster') # ggplot
        sns.lineplot(data=T[T.conn_type==ct],x='numtessels',hue='model',y='Rout',err_style='bars', markers='o', palette='rocket')
        plt.ylim(0.08,.4)
        plt.xticks([80,304,670,1190,1848])
        plt.title(ct)
    pass
    plt.savefig('FigS1.pdf')
    plt.savefig('FigS1.svg')



def plot_scaling(atlas='tessels0162', exp='sc1'):
    for i,s in enumerate(const.return_subjs): #
        Xdata = Dataset(exp,'glm7',atlas,s)
        Xdata.load_mat() # Load from Matlab
        X, INFO1 = Xdata.get_data(averaging="sess") # Get numpy
        Q = X.shape[1]
        if i ==0:
            std=np.empty((24,Q))
        std[i,:] = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])

    gii,name = data.convert_cortex_to_gifti(std.mean(axis=0),atlas=atlas)
    atl_dir = const.base_dir / 'sc1' / 'surfaceWB' / 'group32k'
    surf = []
    surf.append(str(atl_dir / 'fs_LR.32k.L.very_inflated.surf.gii'))
    surf.append(str(atl_dir  / 'fs_LR.32k.R.very_inflated.surf.gii'))
    gdata = gii[0].agg_data()
    view =  nip.view_surf(surf[0],surf_map=gdata,
                vmin=0,vmax=np.max(gdata[np.logical_not(np.isnan(gdata))]),cmap='hot',symmetric_cmap=False)
    return view

def sim_cortex_differences(P=2000,atlas='tessels0162',
                    sigma=2.0,conn_type='one2one'):
    #  alphaR = validate_hyper(X,Y,model.L2regression)
    D=pd.DataFrame()
    for i,s in enumerate(const.return_subjs):
        X1,X2,I1,I2 = getX_cortex(atlas,s)
        N1,Q = X1.shape
        N2,_ = X2.shape
        W = getW(P,Q,conn_type)
        Y1  = X1 @ W.T + np.random.normal(0,sigma,(N1,P))

        MOD =[];
        MOD.append(model.L2regression(alpha=np.exp(3)))
        MOD.append(model.Lasso(alpha=np.exp(-1)))

        for m in range(len(MOD)):
            MOD[m].fit(X1,Y1)
        if i ==0:
            correct=np.empty((24,Q))
            area=np.empty((24,Q))

        numsim=W.sum(axis=0) # Simulations per cortical parcels
        conn = W.T @ (np.abs(MOD[1].coef_)>0)
        correct[i,:] = np.diag(conn)/numsim
        area[i,:] = conn.sum(axis=1)/numsim

    gii,name = data.convert_cortex_to_gifti(area.mean(axis=0),atlas=atlas)
    atl_dir = const.base_dir / 'sc1' / 'surfaceWB' / 'group32k'
    surf = []
    surf.append(str(atl_dir / 'fs_LR.32k.L.very_inflated.surf.gii'))
    surf.append(str(atl_dir  / 'fs_LR.32k.R.very_inflated.surf.gii'))
    gdata = gii[0].agg_data()
    view =  nip.view_surf(surf[0],surf_map=gdata,
                vmin=0,vmax=np.max(gdata[np.logical_not(np.isnan(gdata))]),cmap='hot',symmetric_cmap=False)
    return view

def sim_mappings(type='iid'):
    """
        Explore influence and dectabaility of different mappings between cortex and cerebellum
    """
    P=50
    Q=50
    K=5
    N=4
    X1, X2 = getX_clusters(N,Q,K,eps=0.3)
    fig = plt.figure()
    ax = fig.add_subplot(2,3,1,projection='3d')
    ax.scatter(X1[0,:],X1[1,:],X1[2,:])
    if type=='iid':
        W = np.eye(Q)
    if type=='weight':
        sizeP = np.array([50,0,0,0,0])
        sizeQ = np.array([10,10,10,10,10])
        W = zeros((P,Q))
        q = 0
        p = 0
        for k in range(K):
            W[p:p+sizeP[k], q:q+sizeQ[k]]=getW(sizeP[k],sizeQ[k],'sparse',sparse_num=1)
            p+=sizeP[k]
            q+=sizeQ[k]
    if type=='mix':
        pass
    if type=='mixweight':
        pass
    Y1 = X1 @ W
    X1 = X1 - X1.mean(axis=0)
    Y1 = Y1 - Y1.mean(axis=0)
    G1=X1@X1.T / Q
    G2=Y1@Y1.T / Q
    alpha = cosang(G1,G2)

    # MDS Plot of first matrix 
    V1,lam=pcm.util.classical_mds(G1)
    ax = fig.add_subplot(2,3,2)
    sns.scatterplot(x=V1[:,0],y=V1[:,1],hue=np.arange(N),legend=False)
    ax.axis('equal')

    # MDS Plot of second matrix 
    V2,lam=pcm.util.classical_mds(G2,align=V1)
    ax = fig.add_subplot(2,3,3)
    sns.scatterplot(x=V2[:,0],y=V2[:,1],hue=np.arange(N),legend=False)
    ax.axis('equal')
    ax.set_title(f'Alpha = {alpha:.2f}')

    pass


def cosang(G1,G2):
    cosang=sum(G1*G2)/sqrt(sum(G1*G1)*sum(G2*G2))
    return cosang

def mmd(X,Y,kernel='ip'): 
    """Calculate maximum mean divergence, based on a specific kernel 

    Args:
        X (M x Q ndarray): sample 1
        Y (N x Q ndarray): sample 2
    """
    

if __name__ == "__main__":
    # plot_sim_scenario2()
    # ass
    # sim_cortex_differences()
    # sim_scenario1()
    # sim_cortex_differences()
    sim_mappings(type='weight')
