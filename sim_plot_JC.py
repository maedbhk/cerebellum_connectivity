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
#import timeit

# plotting all the plots together at one time
def plotting(estim_R=20, sde=0, true_rank=1, lambda_list=np.array([0, 0.1,0.2, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2]), filename='_overlambda1_10'):
    
    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    rankname="_rank"+str(true_rank)
    
    sing_val_C_hat_overlambda=pd.read_csv("sing_val_C_hat"+filename+errname+rankname+".csv").iloc[:,1:]
    s_trueC_overlambda=pd.read_csv("s_trueC"+filename+errname+rankname+".csv").iloc[:,1:]
    C_est_err_overlambda=pd.read_csv("C_est_err"+filename+errname+rankname+".csv").iloc[:,1:]
    predic_err_ontest_overlambda=pd.read_csv("predic_err_ontest"+filename+errname+rankname+".csv").iloc[:,1:]
    predic_err_ontraining_overlambda=pd.read_csv("predic_err_ontraining"+filename+errname+rankname+".csv").iloc[:,1:]
    diff_u_overlambda=pd.read_csv("diff_u"+filename+errname+rankname+".csv").iloc[:,1:]
    diff_v_overlambda=pd.read_csv("diff_v"+filename+errname+rankname+".csv").iloc[:,1:]
    converg_overlambda=pd.read_csv("converg"+filename+errname+rankname+".csv").iloc[:,1:]
    diff_sing_val_Chat_C_overlambda=sing_val_C_hat_overlambda-s_trueC_overlambda
    
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
    axes.plot(x_axis.T, s_trueC_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('Singular values of true $C$ vs rank')
    axes.legend(labels=labels)
    plt.show

    #plot the difference between the singular values of \hat{C} and C
    fig, axes = plt.subplots(figsize=(12,9))        
    axes.plot(x_axis.T, diff_sing_val_Chat_C_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('Difference of singular values betwen $\hat{C}$ and true $C$ vs rank')
    axes.legend(labels=labels)
    plt.show

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

    #plot the norm of difference between abs(u-u.hat) vs rank
    fig, axes = plt.subplots(figsize=(12,9))     
    axes.plot(x_axis.T, diff_u_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('$||u_r-\hat{u}_{r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()

    #plot the norm of difference between abs(v-v.hat) vs rank
    fig, axes = plt.subplots(figsize=(12,9))     
    axes.plot(x_axis.T, diff_v_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    plt.title('$||v_r-\hat{v}_{r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()

    print(converg_overlambda)

def plotting_singular_Chat(estim_R=20, sde=0.3, true_rank=1, lambda_list=np.array([1,  2,  3,  4, 5,  6,  7,  8,  9,  10]), filename='_overlambda1_10'):
    
    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    rankname="_rank"+str(true_rank)
    
    sing_val_C_hat_overlambda=pd.read_csv("sing_val_C_hat"+filename+errname+rankname+".csv").iloc[:,1:]
    
    #ploting result for the mean values for all simulations
    labels=[]
    for lambda1 in lambda_list:
        labels.append(f'$log\lambda$= {lambda1}')

    x_axis=np.array([np.arange(1,1+estim_R, 1),]*int(len(lambda_list))) #generate x-axis values: 1, 2,..., estim_R 

    #plot the singular values for C hat vs rank
    fig, axes = plt.subplots(figsize=(6,4.5))        
    axes.plot(x_axis.T, sing_val_C_hat_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    #plt.title('Singular values of $\hat{C}$ vs rank')
    axes.legend(labels=labels)
    plt.show
    
def plotting_diff_norm_C_Chat(estim_R=20, sde=0.3, true_rank=1, lambda_list=np.array([1,  2,  3,  4, 5,  6,  7,  8,  9,  10]), filename='_overlambda1_10'):
    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    rankname="_rank"+str(true_rank)
    
    C_est_err_overlambda=pd.read_csv("C_est_err"+filename+errname+rankname+".csv").iloc[:,1:]
    
    #ploting result for the mean values for all simulations
    labels=[]
    for lambda1 in lambda_list:
        labels.append(f'$\lambda$= {lambda1}')

    x_axis=np.array([np.arange(1,1+estim_R, 1),]*int(len(lambda_list))) #generate x-axis values: 1, 2,..., estim_R 

    #plot the cumulative estimation error||C-C.hat||_F vs rank
    fig, axes = plt.subplots(figsize=(6,4.5))     
    axes.plot(x_axis.T, C_est_err_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    #plt.title('Cumulative estimation error $||C-\hat{C}_{1:r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()
    
def plotting_predic_testset(estim_R=20, sde=0.3, true_rank=1, lambda_list=np.array([1,  2,  3,  4, 5,  6,  7,  8,  9,  10]), filename='_overlambda1_10'):
    
    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    rankname="_rank"+str(true_rank)
    
    predic_err_ontest_overlambda=pd.read_csv("predic_err_ontest"+filename+errname+rankname+".csv").iloc[:,1:]
    
    #ploting result for the mean values for all simulations
    labels=[]
    for lambda1 in lambda_list:
        labels.append(f'$\lambda$= {lambda1}')

    x_axis=np.array([np.arange(1,1+estim_R, 1),]*int(len(lambda_list))) #generate x-axis values: 1, 2,..., estim_R 
    
    #plot the cumulative prediction error on test set||Y-XC.hat||_F vs rank
    fig, axes = plt.subplots(figsize=(6,4.5))     
    axes.plot(x_axis.T, predic_err_ontest_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    #plt.title('Cumulative prediction error on test set $||Y-X\hat{C}_{1:r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()

def plotting_predic_trainingset(estim_R=20, sde=0.3, true_rank=1, lambda_list=np.array([1,  2,  3,  4, 5,  6,  7,  8,  9,  10]), filename='_overlambda1_10'):
    
    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    rankname="_rank"+str(true_rank)
    
    predic_err_ontraining_overlambda=pd.read_csv("predic_err_ontraining"+filename+errname+rankname+".csv").iloc[:,1:]
    
    #ploting result for the mean values for all simulations
    labels=[]
    for lambda1 in lambda_list:
        labels.append(f'$\lambda$= {lambda1}')

    x_axis=np.array([np.arange(1,1+estim_R, 1),]*int(len(lambda_list))) #generate x-axis values: 1, 2,..., estim_R 
    
    #plot the cumulative prediction error on training set||Y-XC.hat||_F vs rank
    fig, axes = plt.subplots(figsize=(6,4.5))     
    axes.plot(x_axis.T, predic_err_ontraining_overlambda.T) #generate plot for multiple ranks
    plt.xlabel('r (Rank)')
    plt.ylabel('')
    #plt.title('Cumulative prediction error on training set $||Y-X\hat{C}_{1:r}||_F$ of vs rank')
    axes.legend(labels=labels)
    plt.show()
