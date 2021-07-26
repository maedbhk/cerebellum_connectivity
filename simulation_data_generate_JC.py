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

import Class_SmoothLS_JC 
#function generating the matrix C by first generating u's and v's, then implementing the low rank decomposition
def generate_C(rank, nrow, ncol):
    #nrow: the number of cortex voxels
    #ncol: the number of cerebellum voxels
    leng_sqr_cortex=int(math.sqrt(nrow))
    leng_sqr_cere=int(math.sqrt(ncol))
    
   
    C=np.zeros((nrow, ncol))
    betas=np.zeros(rank*(nrow+ncol))
    U=np.zeros((nrow, rank))
    V=np.zeros((ncol, rank))
    
    for r in range(rank):
        #generating the spatial smoothed cortex vector u_r
        cortex_mat = gaussian_filter(np.random.normal(0,1,(leng_sqr_cortex,leng_sqr_cortex)),2)
        #vectorize cortex_mat by rows
        cortex_u=cortex_mat.reshape(-1) 
        #betas[r*nrow:((r+1)*nrow)]=cortex_u
        U[:,r]=cortex_u
        
        #generating the spatial smoothed cerebellum vector v_r

        cere_mat = gaussian_filter(np.random.normal(0,1,(leng_sqr_cere,leng_sqr_cere)),2)
        #vectorize cere_mat by rows
        cere_v=cere_mat.reshape(-1)      
        #betas[(rank*nrow+r*ncol):(rank*nrow+((r+1)*ncol))]=cere_v
        V[:,r]=cere_v
        
        C=C+np.outer(cortex_u,cere_v)
        
    return C, U, V

def generate_X(n_obs, num_vox_cortex):
    """
        Generate X matrix with each row be the vectorized smoothness gaussian.
    """
    
    X=np.zeros((n_obs,num_vox_cortex)) 
    leng_sqr_cortex=int(math.sqrt(num_vox_cortex))
#    print(leng_sqr_cortex)
    
    for row in range(n_obs):
        #generate each row of X (smoothed)
        Z=np.random.normal(0,1,(leng_sqr_cortex,leng_sqr_cortex))
        cortex_mat = gaussian_filter(Z,2)
        #vectorize cortex_mat by rows
        X[row,:]=cortex_mat.reshape(-1)
        
    return X

from scipy.spatial import distance_matrix

def distance_mat(side_length):
    
    #side_length: the length of one side of the square
    
    x = np.arange(1,1+side_length, 1)
    
    xx, yy = np.meshgrid(x, x, sparse=False)
    
    #x_coor=pd.DataFrame(xx.reshape(-1,1))
  
    #y_coor=pd.DataFrame(yy.reshape(-1,1))
   
    coordi=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
 
    return distance_matrix(coordi,coordi)


#Simulation function
def sim_fun(C, estim_R=20, num_vox_cortex=400, num_vox_cere=100, n_obs=40, sd_err=0.3, lambda_input=1):
    '''
    true_R: rank of the true pattern
    estim_R: rank of the estimated pattern
    num_vox_cortex: the number of voxels in cortex
    num_vox_cere: the number of voxels in cerebellum
    n_obs: the number of observations
    lambda is set to be 1
    C: the true C
    U: the true U used for constructing the C
    V: the true V used for constructing the C
    '''

    #Step 1. Generate data 
    X_train=generate_X(n_obs, num_vox_cortex) #generate the training covariate martix X_train
    #C, U, V=generate_C(true_R,num_vox_cortex,num_vox_cere) # generate true coefficient matrix C for each simulation along 
                                                           # with the correpsonding singular vectors
    X_dist_mat=distance_mat(math.sqrt(num_vox_cortex)) #generate distance matrix for cortex (400*400)
    Y_dist_mat=distance_mat(math.sqrt(num_vox_cere)) # generate distance matrix for cerebellumn (100*100)
    E_train=np.random.normal(loc=0.0, scale=sd_err,size=(n_obs, num_vox_cere)) #generate the error matrix E for training set
                                                                            # from iid N(0,sd_err)
    Y_train=X_train@C+E_train #construct the response matrix for training set Y_train
    
    u, s, vh=LA.svd(C) #obtain the SVD to true C
    #print(f'The singular valeus of the true C (first 20 values) are: {s[0:estim_R]}')
    U_C=u[:,0:estim_R] #store the sigular vector u's
    V_C=vh[0:estim_R,:].T #store the singular vector v's
    
    #Step 2. Using iterative method to estimate C
    ##where rank  of C is set to be 10, lambda=1
    Fit_training=Class_SmoothLS_JC.LowRankSmoothLS(rank=estim_R, lambda1=lambda_input) # estim_R is the estimated rank predefined, here we may want to specify
                                            # a larger estim_R to clearly see if the rank trend at the true rank point
    Fit_training.fit_iterative2(X_train, Y_train, X_dist_mat, Y_dist_mat) #implement the iterative fitting
    
    #Generate testing set used for evaluation
    X_test=generate_X(n_obs, num_vox_cortex) #generate test X_test
    E_test=np.random.normal(loc=0.0, scale=sd_err,size=(n_obs, num_vox_cere)) #generate test error matrix E_test
    Y_test=X_test@C+E_test
        
    #Step 3. Evaluate the performance of estimation and store the calculation at each component
    sing_val_Chat=np.zeros(estim_R) #store the singular values of the estimated C
    C_est_err=np.zeros(estim_R) #store the cumulative estimation error
    predic_err_ontest=np.zeros(estim_R) #store the cumulative prediction error on test set
    predic_err_ontraining=np.zeros(estim_R) #store the cumulative prediction error on training set
    diff_u=np.zeros(estim_R)
    diff_v=np.zeros(estim_R)
    for i in range(estim_R):
        #calculate the singular values of the estimated C for each component
        sing_val_Chat[i]=LA.norm(Fit_training.U[:,i])*LA.norm(Fit_training.V[:,i])
        #calculate the cumulative estimated C
        Cumu_Ci=Fit_training.U[:,0:(i+1)]@Fit_training.V[:,0:(i+1)].T
        #calculate the cumulative estimation error
        C_est_err[i]=LA.norm(C-Cumu_Ci)
        #calculate the cumulative prediction error
        predic_err_ontest[i]=LA.norm(Y_test-X_test@Cumu_Ci)
        #calculate the prediction error on training set
        predic_err_ontraining[i]=LA.norm(Y_train-X_train@Cumu_Ci)
        #calculate the difference of u's for each component
        diff_u[i]=LA.norm(abs(U_C[:,i]/LA.norm(U_C[:,i]))-abs(Fit_training.U[:,i]/LA.norm(Fit_training.U[:,i])))
        #calculate the difference of v's for each component
        diff_v[i]=LA.norm(abs(V_C[:,i]/LA.norm(V_C[:,i]))-abs(Fit_training.V[:,i]/LA.norm(Fit_training.V[:,i])))
            
        
#    #plot the values in the same graph
#    fig, axes = plt.subplots(figsize=(10,7))
#    x_axis=np.arange(1,1+estim_R, 1) #generate x-axis values: 1, 2,..., estim_R
#    plt.plot(x_axis, sing_val_Chat, color ="red", label="Singular values of $\hat{C}$") 
#    plt.plot(x_axis, s[0:estim_R], color ="orange", label="Singular values of true C") 
#    plt.plot(x_axis, C_est_err, color ="blue", label="Cumulative estimation error") 
#    plt.plot(x_axis, predic_err_ontest, color ="yellow", label="Cumulative prediction error")
#    #plt.plot(x_axis, diff_u, color ="pink", label="Estimation error for $u_r$")
#    #plt.plot(x_axis, diff_v, color ="green", label="Estimation error for $v_r$")
#    plt.xlabel('Rank')
#    plt.ylabel('')
#    plt.title('Errors vs rank')
#    plt.legend()
#    plt.show()

    s_trueC=np.zeros(estim_R)+s[0:estim_R]
   
    return sing_val_Chat, s_trueC, C_est_err, predic_err_ontest, predic_err_ontraining, diff_u, diff_v, Fit_training.converge
    #return (1). the singular values of the estimated coefficient matrix C hat (a vector of lenght estim_R)
    #       (2). the singular values of the true C (a vector of lenght estim_R)
    #       (3). rhe cumulative estimation error for C (a vector of lenght estim_R)
    #       (4). the cumulative prediction error for Y (a vector of lenght estim_R)
    #       (5). the norm of the difference between true u_r and u_r hat for each component (a vector of lenght estim_R)
    #       (6). the norm of the difference between true v_r and v_r hat for each component (a vector of lenght estim_R)
    #       (7). if this simulation result converge or not (True/False)
    

#    #decide lambda by 10-fold cross-validation
#    num_fold=10
#    kf=KFold(n_splits=num_fold, shuffle=True)
#    
#    print("Start cross-validation for searching lambda:")
#    i=0
#    Ypred_sum_error=np.zeros(6)
#    for lambda_ in np.linspace(start=0.5,stop=3,num=6):
#        print('cv lambda:')
#        print(lambda_)
#        #obtaining the sum of squared residuals for the sequence of lambda
#        for train_ind, test_ind in kf.split(Y):
#            Fit=LowRankSmoothLS(rank=R, lambda1=lambda_)
#            Fit.fit_iterative(X[train_ind], Y[train_ind], X_dist_mat, Y_dist_mat)
#            Ypred_sum_error[i]=Ypred_sum_error[i]+LA.norm(Y[test_ind]-X[test_ind]@Fit.C)**2
#        i+=1    
#    lambda1_cv=np.linspace(start=0,stop=2,num=21)[list(Ypred_sum_error).index(min(Ypred_sum_error))]
#    print('The CV lambda value is:')
#    print(lambda1_cv)   
    
#Simulation function
def sim_fun(C, smooth_para, estim_R=20, num_vox_cortex=400, num_vox_cere=100, n_obs=40, sd_err=0.3, lambda_input=1):
    '''
    true_R: rank of the true pattern
    estim_R: rank of the estimated pattern
    num_vox_cortex: the number of voxels in cortex
    num_vox_cere: the number of voxels in cerebellum
    n_obs: the number of observations
    lambda is set to be 1
    C: the true C
    U: the true U used for constructing the C
    V: the true V used for constructing the C
    '''

    #Step 1. Generate data 
    X_train=generate_X(n_obs, num_vox_cortex,smooth_para) #generate the training covariate martix X_train
    #C, U, V=generate_C(true_R,num_vox_cortex,num_vox_cere) # generate true coefficient matrix C for each simulation along 
                                                           # with the correpsonding singular vectors
    X_dist_mat=distance_mat(math.sqrt(num_vox_cortex)) #generate distance matrix for cortex (400*400)
    Y_dist_mat=distance_mat(math.sqrt(num_vox_cere)) # generate distance matrix for cerebellumn (100*100)
    E_train=np.random.normal(loc=0.0, scale=sd_err,size=(n_obs, num_vox_cere)) #generate the error matrix E for training set
                                                                            # from iid N(0,sd_err)
    Y_train=X_train@C+E_train #construct the response matrix for training set Y_train

    
    #Step 2. Using iterative method to estimate C
    ##where rank  of C is set to be 10, lambda=1
    Fit_training=LowRankSmoothLS(rank=estim_R, lambda1=lambda_input) # estim_R is the estimated rank predefined, here we may want to specify
                                            # a larger estim_R to clearly see if the rank trend at the true rank point
    Fit_training.fit_iterative2(X_train, Y_train, X_dist_mat, Y_dist_mat) #implement the iterative fitting
    
    #Generate testing set used for evaluation
    X_test=generate_X(n_obs, num_vox_cortex,smooth_para) #generate test X_test
    E_test=np.random.normal(loc=0.0, scale=sd_err,size=(n_obs, num_vox_cere)) #generate test error matrix E_test
    Y_test=X_test@C+E_test
        
    #Step 3. Evaluate the performance of estimation and store the calculation at each component
    sing_val_Chat=np.zeros(estim_R) #store the singular values of the estimated C
    C_est_err=np.zeros(estim_R) #store the cumulative estimation error
    predic_err_ontest=np.zeros(estim_R) #store the cumulative prediction error on test set
    predic_err_ontraining=np.zeros(estim_R) #store the cumulative prediction error on training set

    for i in range(estim_R):
        #calculate the singular values of the estimated C for each component
        sing_val_Chat[i]=LA.norm(Fit_training.U[:,i])*LA.norm(Fit_training.V[:,i])
        #calculate the cumulative estimated C
        Cumu_Ci=Fit_training.U[:,0:(i+1)]@Fit_training.V[:,0:(i+1)].T
        #calculate the cumulative estimation error
        C_est_err[i]=LA.norm(C-Cumu_Ci)
        #calculate the cumulative prediction error
        predic_err_ontest[i]=LA.norm(Y_test-X_test@Cumu_Ci)
        #calculate the prediction error on training set
        predic_err_ontraining[i]=LA.norm(Y_train-X_train@Cumu_Ci)


   
    return sing_val_Chat, C_est_err, predic_err_ontest, predic_err_ontraining
    #return 
    #       
    #       (1). rhe cumulative estimation error for C (a vector of lenght estim_R)
    #       (2). the cumulative prediction error for Y on test set (a vector of lenght estim_R)
    #       (3). the cumulative prediction error for Y on trsining set(a vector of lenght estim_R)
    #       (4). if this simulation result converge or not (True/False)



