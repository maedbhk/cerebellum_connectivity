
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
import Class_SmoothLS_JC 
import simulation_data_generate_JC as sum_fun #impoer simulation functions
import sim_plot_JC as sp #import plotting functions

import connectivity.model as model # import models
import connectivity.evaluation as ev # import evaluation methods



simu_num=50
estim_R=16
sde=0.1
#lambda_list=np.array([1,  2,  3,  4, 5,  6,  7,  8,  9,  10])
lambda_list=np.array([10**-1, 1, 10, 10**2, 10**3])
len_lambda=len(lambda_list)


for i in range(10): #iterate the true R of C from 1 to 10
    print(f'True rank: {i+1}')
    
    trueC, U, V=sum_fun.generate_C(rank=i+1, nrow=400, ncol=100)
    
    sing_val_C_hat_overlambda=np.zeros((len_lambda, estim_R))
    s_trueC_overlambda=np.zeros((len_lambda, estim_R))
    C_est_err_overlambda=np.zeros((len_lambda, estim_R))
    predic_err_ontest_overlambda=np.zeros((len_lambda, estim_R))
    predic_err_ontraining_overlambda=np.zeros((len_lambda, estim_R))
    diff_u_overlambda=np.zeros((len_lambda, estim_R))
    diff_v_overlambda=np.zeros((len_lambda, estim_R))
    converg_overlambda=np.zeros(len_lambda)
    
    ind=0
    for lambda1 in lambda_list:
        print(f'$\lambda$ value:{lambda1}')
        sing_val_C_hat=np.zeros((simu_num, estim_R))
        s_trueC=np.zeros((simu_num, estim_R))
        C_est_err=np.zeros((simu_num, estim_R))
        predic_err_ontest=np.zeros((simu_num, estim_R))
        predic_err_ontraining=np.zeros((simu_num, estim_R))
        diff_u=np.zeros((simu_num, estim_R))
        diff_v=np.zeros((simu_num, estim_R))
        converg=np.zeros(simu_num)
        
        for j in range(simu_num): 
            
            print(f'Simulation number: {j+1}')
            sing_val_C_hat[j],s_trueC[j],C_est_err[j],predic_err_ontest[j], predic_err_ontraining[j],diff_u[j],diff_v[j], converg[j]=sum_fun.sim_fun(C=trueC,estim_R=estim_R,sd_err=sde,lambda_input=lambda1)
    
        
        sing_val_C_hat_overlambda[ind]=np.mean(sing_val_C_hat, axis=0)
        s_trueC_overlambda[ind]=np.mean(s_trueC, axis=0)
        C_est_err_overlambda[ind]=np.mean(C_est_err, axis=0)
        predic_err_ontest_overlambda[ind]=np.mean(predic_err_ontest, axis=0)
        predic_err_ontraining_overlambda[ind]=np.mean(predic_err_ontraining, axis=0)
        diff_u_overlambda[ind]=np.mean(diff_u, axis=0)
        diff_v_overlambda[ind]=np.mean(diff_v, axis=0)
        converg_overlambda[ind]=np.mean(converg, axis=0)
        
        ind+=1

    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    rankname="_rank"+str(i+1)
    #store the average simulation result for all ranks 
    pd.DataFrame(sing_val_C_hat_overlambda).to_csv("sing_val_C_hat_overlambda_logscale"+errname+rankname+".csv") #row is true rank index; column is estimated rank index
    pd.DataFrame(s_trueC_overlambda).to_csv("s_trueC_overlambda_logscale"+errname+rankname+".csv")
    pd.DataFrame(C_est_err_overlambda).to_csv("C_est_err_overlambda_logscale"+errname+rankname+".csv")
    pd.DataFrame(predic_err_ontest_overlambda).to_csv("predic_err_ontest_overlambda_logscale"+errname+rankname+".csv")
    pd.DataFrame(predic_err_ontraining_overlambda).to_csv("predic_err_ontraining_overlambda_logscale"+errname+rankname+".csv")
    pd.DataFrame(diff_u_overlambda).to_csv("diff_u_overlambda_logscale"+errname+rankname+".csv")
    pd.DataFrame(diff_v_overlambda).to_csv("diff_v_overlambda_logscale"+errname+rankname+".csv")
    pd.DataFrame(converg_overlambda).to_csv("converg_overlambda_logscale"+errname+rankname+".csv")
    
    sp.plotting(estim_R=estim_R, sde=sde, true_rank=i+1, lambda_list=np.log10(lambda_list),filename='_overlambda_logscale')