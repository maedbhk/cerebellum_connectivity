import numpy as np
import pandas as pd

from numpy import linalg as LA #this package is used when calculating the norm

import simudata_generator2_JC as sg2
import plotting_simuresults_smooth_JC as plt_sim

#Basix settings for the simulation
simu_num=50 #number of runs for each case (each case means setting specific sde, smooth_index, lambda)
estim_R=20 # define the estimated rank 
rank=5 #define the true rank of true C

#varying factors
#(1).
sde=0.1 #standard deviation to generate the simulated data
#(2).
smooth_index=np.array([1,2,3,4]) #the smooth index which setting the gaussian filter smoothness for both C and X
#(3). lambda 
lambda_list=np.array([10**-1, 1, 10, 50, 10**2, 10**3]) # the list of lambda values used when fitting the data
len_lambda=len(lambda_list)


for i in range(len(smooth_index)): #loop among the smooth levels
    
    smooth=smooth_index[i]
    print(f'Smooth parameter: {smooth_index[i]}')
    
    trueC, U, V=sg2.generate_C(rank=rank, nrow=400, ncol=100,smooth_para=smooth) 
    u, s, vh=LA.svd(trueC) #obtain the SVD to true C
    s_trueC=s[0:estim_R]
    
    sing_val_C_hat_overlambda=np.zeros((len_lambda, estim_R))
    C_est_err_overlambda=np.zeros((len_lambda, estim_R))
    predic_err_ontest_overlambda=np.zeros((len_lambda, estim_R))
    predic_err_ontraining_overlambda=np.zeros((len_lambda, estim_R))

    
    ind=0
    for lambda1 in lambda_list:
        print(f'$\lambda$ value:{lambda1}')
        sing_val_C_hat=np.zeros((simu_num, estim_R))
        C_est_err=np.zeros((simu_num, estim_R))
        predic_err_ontest=np.zeros((simu_num, estim_R))
        predic_err_ontraining=np.zeros((simu_num, estim_R))

        
        for j in range(simu_num): 
            
            print(f'Simulation number: {j+1}')
            sing_val_C_hat[j],C_est_err[j],predic_err_ontest[j], predic_err_ontraining[j]=sg2.sim_fun(C=trueC,smooth_para=smooth,estim_R=estim_R,sd_err=sde,lambda_input=lambda1)
        
        sing_val_C_hat_overlambda[ind]=np.mean(sing_val_C_hat, axis=0)
        C_est_err_overlambda[ind]=np.mean(C_est_err, axis=0)
        predic_err_ontest_overlambda[ind]=np.mean(predic_err_ontest, axis=0)
        predic_err_ontraining_overlambda[ind]=np.mean(predic_err_ontraining, axis=0)
        
        
        ind+=1

    errname="_sderr"+str(sde)
    errname=errname.replace('.','')
    smoothname="_smooth"+str(smooth)
    #store the average simulation result for all ranks 
    pd.DataFrame(sing_val_C_hat_overlambda).to_csv("sing_val_C_hat_overlambda"+errname+smoothname+".csv") 
    pd.DataFrame(s_trueC).to_csv("singularValu_trueC"+errname+smoothname+".csv")
    pd.DataFrame(C_est_err_overlambda).to_csv("C_est_err_overlambda"+errname+smoothname+".csv")
    pd.DataFrame(predic_err_ontest_overlambda).to_csv("predic_err_ontest_overlambda"+errname+smoothname+".csv")
    pd.DataFrame(predic_err_ontraining_overlambda).to_csv("predic_err_ontraining_overlambda"+errname+smoothname+".csv")

    
    #plt_sim.plotting(estim_R=estim_R, sde=sde, smooth_para=smooth, lambda_list=lambda_list,filename='_overlambda')