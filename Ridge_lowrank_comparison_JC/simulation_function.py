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
from data_generate import generate_X, distance_mat
from LowRankSmoothLS import LowRankSmoothLS

#Simulation function
def sim_fun(C, estim_R=20, num_vox_cortex=100, num_vox_cere=400, n_obs=40, sd_E=0.3, lambda1_input=1, lambda2_input=1):
    """
        true_R: rank of the true pattern
        estim_R: rank of the estimated pattern
        num_vox_cortex: the number of voxels in cortex
        num_vox_cere: the number of voxels in cerebellum
        n_obs: the number of observations
        lambda1 is set to be 1
        lambda2 is set to be 1
        C: the true C
        U: the true U used for constructing the C
        V: the true V used for constructing the C
    """
    #Step 1. Generate data 
    X_train=generate_X(n_obs, num_vox_cortex, smooth_level=2) #generate the training covariate martix X_train
    #which is the same for all cases
    E_train=np.random.normal(loc=0.0, scale=sd_E,size=(n_obs, num_vox_cere)) #generate the error matrix E for training set
                                                                            # from iid N(0,sd_err)     
    Y_train=X_train@C+E_train #construct the response matrix for training set Y_train

    X_dist_mat=distance_mat(math.sqrt(num_vox_cortex)) #generate distance matrix for cortex (100*100)
    Y_dist_mat=distance_mat(math.sqrt(num_vox_cere)) # generate distance matrix for cerebellumn (400*400)

    #u, s, vh=LA.svd(C) #obtain the SVD to true C
    #print(f'The singular valeus of the true C (first 20 values) are: {s[0:estim_R]}')
    #U_C=u[:,0:estim_R] #store the sigular vector u's
    #V_C=vh[0:estim_R,:].T #store the singular vector v's

    #Step 2. Using iterative method to estimate C
    ##where rank  of C is set to be 10, lambda=1
    Fit_training=LowRankSmoothLS(rank=estim_R, lambda1=lambda1_input, lambda2=lambda2_input) # estim_R is the estimated rank predefined, here we may want to specify
                                            # a larger estim_R to clearly see if the rank trend at the true rank point
    Fit_training.fit_iterative2(X_train, Y_train, X_dist_mat, Y_dist_mat) #implement the iterative fitting

    #Generate testing set used for evaluation
    X_test=generate_X(n_obs, num_vox_cortex, smooth_level=2) #generate test X_test
    E_test=np.random.normal(loc=0.0, scale=sd_E,size=(n_obs, num_vox_cere)) #generate test error matrix E_test
    Y_test=X_test@C+E_test

    #Step 3. Evaluate the performance of estimation and store the calculation at each component
    #sing_val_Chat=np.zeros(estim_R) #store the singular values of the estimated C
    C_est_err=np.zeros(estim_R) #store the cumulative estimation error
    predic_err_ontest=np.zeros(estim_R) #store the cumulative prediction error on test set
    #predic_err_ontraining=np.zeros(estim_R) #store the cumulative prediction error on training set
    #diff_u=np.zeros(estim_R)
    #diff_v=np.zeros(estim_R)
    for i in range(estim_R):
        #calculate the singular values of the estimated C for each component
        #sing_val_Chat[i]=LA.norm(Fit_training.U[:,i])*LA.norm(Fit_training.V[:,i])
        #calculate the cumulative estimated C
        Cumu_Ci=Fit_training.U[:,0:(i+1)]@Fit_training.V[:,0:(i+1)].T
        #calculate the cumulative estimation error
        C_est_err[i]=LA.norm(C-Cumu_Ci)
        #calculate the cumulative prediction error
        predic_err_ontest[i]=LA.norm(Y_test-X_test@Cumu_Ci)
        #calculate the prediction error on training set
        #predic_err_ontraining[i]=LA.norm(Y_train-X_train@Cumu_Ci)
        #calculate the difference of u's for each component
        #diff_u[i]=LA.norm(abs(U_C[:,i]/LA.norm(U_C[:,i]))-abs(Fit_training.U[:,i]/LA.norm(Fit_training.U[:,i])))
        #calculate the difference of v's for each component
        #diff_v[i]=LA.norm(abs(V_C[:,i]/LA.norm(V_C[:,i]))-abs(Fit_training.V[:,i]/LA.norm(Fit_training.V[:,i])))

    C_est_err_min=np.amin(C_est_err)
    numcomp_C=np.where(C_est_err==C_est_err_min)
    pred_err_min=np.amin(predic_err_ontest)
    numcomp_pred=np.where(predic_err_ontest==pred_err_min)

    return C_est_err_min, pred_err_min, numcomp_pred[0][0]+1
    #return
    #       (3). rhe cumulative estimation error for C (a vector of lenght estim_R)\n",
    #       (4). the cumulative prediction error for Y on the test set (a vector of lenght estim_R)"