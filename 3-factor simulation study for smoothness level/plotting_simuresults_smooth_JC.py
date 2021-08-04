import numpy as np
import pandas as pd
from scipy.optimize import minimize #this package is used when finding the minimizer of the objective
from numpy import linalg as LA #this package is used when calculating the norm
import seaborn as sns #this package is used when ploting matrix C
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random #set seed
import math
from scipy.spatial import distance_matrix
import statistics

# plotting all the plots together at one time

def plotting(estim_R=20, sde=0.1, smooth_para=1, lambda_list=np.array([0, 0.1,0.2, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2]), filename='_overlambda'):
    
    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    smoothname="_smooth"+str(smooth_para)
    
    sing_val_C_hat_overlambda=pd.read_csv("sing_val_C_hat"+filename+errname+smoothname+".csv").iloc[:,1:]
    s_trueC=pd.read_csv("singularValu_trueC"+errname+smoothname+".csv").iloc[:,1:]
    C_est_err_overlambda=pd.read_csv("C_est_err"+filename+errname+smoothname+".csv").iloc[:,1:]
    predic_err_ontest_overlambda=pd.read_csv("predic_err_ontest"+filename+errname+smoothname+".csv").iloc[:,1:]
    predic_err_ontraining_overlambda=pd.read_csv("predic_err_ontraining"+filename+errname+smoothname+".csv").iloc[:,1:]
    
    #ploting result for the mean values for all simulations
    labels=[]
    for lambda1 in lambda_list:
        labels.append(f'$\lambda$= {lambda1}')

    x_axis=np.array([np.arange(1,1+estim_R, 1),]*int(len(lambda_list))) #generate x-axis values: 1, 2,..., estim_R 

    #plot the singular values for C hat vs rank
    fig, axes = plt.subplots(figsize=(12,9))        
    axes.plot(x_axis.T, sing_val_C_hat_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('Singular values of $\hat{C}$ vs rank')
    axes.legend(labels=labels)
    plt.show

    #plot the singular values for the true C
    fig, axes = plt.subplots(figsize=(12,9))        
    axes.plot(s_trueC) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('Singular values of true $C$ vs rank')
    #axes.legend(labels=labels)
    plt.show

#     #plot the difference between the singular values of \hat{C} and C
#     fig, axes = plt.subplots(figsize=(12,9))        
#     axes.plot(x_axis.T, diff_sing_val_Chat_C_overlambda.T) #generate plot for multiple ranks
#     plt.xlabel('r (Rank)')
#     plt.ylabel('')
#     plt.title('Difference of singular values betwen $\hat{C}$ and true $C$ vs rank')
#     axes.legend(labels=labels)
#     plt.show

    #plot the cumulative estimation error||C-C.hat||_F vs rank
    fig, axes = plt.subplots(figsize=(12,9))     
    axes.plot(x_axis.T, C_est_err_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('Cumulative estimation error $||C-\hat{C}_{1:r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()

    #plot the cumulative prediction error on test set||Y-XC.hat||_F vs rank
    fig, axes = plt.subplots(figsize=(12,9))     
    axes.plot(x_axis.T, predic_err_ontest_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('Cumulative prediction error on test set $||Y-X\hat{C}_{1:r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()

    #plot the cumulative prediction error on training set||Y-XC.hat||_F vs rank
    fig, axes = plt.subplots(figsize=(12,9))     
    axes.plot(x_axis.T, predic_err_ontraining_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('Cumulative prediction error on training set $||Y-X\hat{C}_{1:r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()

