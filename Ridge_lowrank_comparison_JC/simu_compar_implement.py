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
from data_generate import generate_X, generate_C_ridge, generate_C_lowranksmooth, generate_C_lowrank, distance_mat
from LowRankSmoothLS import LowRankSmoothLS
from simulation_function import sim_fun

num_vox_cortex=100
num_vox_cere=400
n_obs=40
simu_num=30
estim_R=10
sd_E=0.3
sd_C=1
lambda1_log_list=np.array([-2,-1,0,1,2,3,4,5])
lambda2_log_list=np.array([-2,-1,0,1,2,3,4,5])
len_lambda1=len(lambda1_log_list)
len_lambda2=len(lambda2_log_list)

num_compC=5 #true number of component for C

#for i in range(num_compC): #iterate the true R of C from 1 to 10
print(f'True number of component: {num_compC}')

trueC, U, V=generate_C_lowranksmooth(rank=num_compC, nrow=100, ncol=400, sd=sd_C, smooth_level=2)

lambda_value_indicator=np.zeros((len_lambda1*len_lambda2,2))
run_results_C_est=np.zeros((len_lambda1*len_lambda2, simu_num))
run_results_pred_err=np.zeros((len_lambda1*len_lambda2, simu_num))
run_results_est_rank=np.zeros((len_lambda1*len_lambda2, simu_num))

lambda_id=0
for log_lambda1 in lambda1_log_list:
    lambda1=np.exp(log_lambda1)
    for log_lambda2 in lambda2_log_list:
        lambda2=np.exp(log_lambda2)
        print(f"lambda1={lambda1}; lambda2={lambda2}")
        lambda_value_indicator[lambda_id,:]=[log_lambda1, log_lambda2]
        for run_id in range(simu_num):
            print(f"Run number: {run_id+1}")
            run_results_C_est[lambda_id,run_id],run_results_pred_err[lambda_id,run_id],\
            run_results_est_rank[lambda_id,run_id]=sim_fun(trueC, estim_R=estim_R, num_vox_cortex=num_vox_cortex, \
                                                           num_vox_cere=num_vox_cere, n_obs=n_obs, sd_E=sd_E, \
                                                           lambda1_input=lambda1, lambda2_input=lambda2)
            run_id+=1
        lambda_id+=1

lambda_value_indicator=pd.DataFrame(lambda_value_indicator)
C_est_result=pd.DataFrame(run_results_C_est)
C_est_result["mean"]=C_est_result.mean(axis=1)
C_est_result=pd.concat([C_est_result,lambda_value_indicator],axis=1)
C_est_result.to_csv("C_est_result_lowrank.csv")

pred_result=pd.DataFrame(run_results_pred_err)
pred_result["mean"]=pred_result.mean(axis=1)
pred_result=pd.concat([pred_result,lambda_value_indicator],axis=1)
pred_result.to_csv("pred_result_lowrank.csv")

comp_result=pd.DataFrame(run_results_est_rank)
comp_result=pd.concat([comp_result,lambda_value_indicator],axis=1)
comp_result.to_csv("comp_result_lowrank.csv")