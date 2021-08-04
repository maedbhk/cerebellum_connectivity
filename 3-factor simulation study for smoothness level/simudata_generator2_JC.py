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
import SmoothLS_Class_JC as SLS


def generate_C(rank, nrow, ncol,smooth_para=2):
    """
    generate_C: function generating the matrix C by layers: 
                first generating u_r and v_r, then using C=sum_(u_r*v_r^T)
    nrow: the number of cortex voxels, should be a squred value, like 6^2, 10^2,...
    ncol: the number of cerebellum voxels, should be a squred value
    """

    leng_sqr_cortex=int(math.sqrt(nrow))
    leng_sqr_cere=int(math.sqrt(ncol))
    
    #initiation
    C=np.zeros((nrow, ncol))
    betas=np.zeros(rank*(nrow+ncol))
    U=np.zeros((nrow, rank))
    V=np.zeros((ncol, rank))
    
    #construct true matrix C by layers
    for r in range(rank):
        #generating the spacial smoothed cortex vector u_r from N(0,1) with smooth parameter "smooth_para"
        cortex_mat = gaussian_filter(np.random.normal(0,1,(leng_sqr_cortex,leng_sqr_cortex)),smooth_para)
        #vectorize cortex_mat by rows
        cortex_u=cortex_mat.reshape(-1) 
        sd_u=statistics.stdev(cortex_u)
        cortex_u_stdiz=cortex_u/sd_u #standardize vector u_r
        U[:,r]=cortex_u_stdiz
        
        #generating the spatial smoothed cerebellum vector v_r from N(0,1) with smooth parameter "smooth_para"
        cere_mat = gaussian_filter(np.random.normal(0,1,(leng_sqr_cere,leng_sqr_cere)),smooth_para)
        #vectorize cere_mat by rows
        cere_v=cere_mat.reshape(-1) 
        sd_v=statistics.stdev(cere_v)
        cere_v_stdize=cere_v/sd_v #standardize vector v_r
        V[:,r]=cere_v_stdize
        
        C=C+np.outer(cortex_u,cere_v)
        
    return C, U, V

def generate_X(n_obs, num_vox_cortex,smooth_para=2):
    """
        generate_X: Generate X matrix (the cortex data) by rows. 
        Each row is the vectorized standardized smoothness gaussian.
        So the std of each row of vectorrized X is 1.

        n_obs: the rows of the X
        num_vox_cortex: the number of columns of X, should be a squared value
    """
    #initilization
    X=np.zeros((n_obs,num_vox_cortex)) 
    leng_sqr_cortex=int(math.sqrt(num_vox_cortex))
#    print(leng_sqr_cortex)
    
    for row in range(n_obs):
        #generate each row of X (smoothed)
        Z=np.random.normal(0,1,(leng_sqr_cortex,leng_sqr_cortex))
        cortex_mat = gaussian_filter(Z,smooth_para)
        row_X=cortex_mat.reshape(-1)
        sd=statistics.stdev(row_X)
        row_X_sdtize=row_X/sd
        #vectorize cortex_mat by rows
        X[row,:]=row_X_sdtize
        
    return X



def distance_mat(side_length):
    
    """
        Generating distance matrix for the data in a square grid with unit 1.
        side_length: the length of one side of the square grid
    """  
    
    x = np.arange(1,1+side_length, 1)
    
    xx, yy = np.meshgrid(x, x, sparse=False)
    
    #x_coor=pd.DataFrame(xx.reshape(-1,1))
  
    #y_coor=pd.DataFrame(yy.reshape(-1,1))
   
    coordi=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
 
    return distance_matrix(coordi,coordi)

    
def sim_fun(C, smooth_para, estim_R=20, num_vox_cortex=400, num_vox_cere=100, n_obs=40, sd_err=0.3, lambda_input=1):
    '''
    Simulation function
    C: the true C
    smooth_para: the smoothness parameter used to set the smoothness level in X
    true_R: true rank of the true pattern C
    estim_R: number of rank in estimation
    num_vox_cortex: the number of voxels in cortex data (X)
    num_vox_cere: the number of voxels in cerebellum data (Y)
    n_obs: the number of observations (the number of rows in X and Y)
    lambda is set to be 1 as default.
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
    Fit_training=SLS.LowRankSmoothLS(rank=estim_R, lambda1=lambda_input) # estim_R is the estimated rank predefined, here we may want to specify
                                            # a larger estim_R to clearly see if the rank trend at the true rank point
    Fit_training.fit_iterative2(X_train, Y_train, X_dist_mat, Y_dist_mat) #implement the iterative fitting
    
    #Generate testing set used for evaluation
    X_test=generate_X(n_obs, num_vox_cortex,smooth_para) #generate test X_test
    E_test=np.random.normal(loc=0.0, scale=sd_err,size=(n_obs, num_vox_cere)) #generate test error matrix E_test
    Y_test=X_test@C+E_test
        
    #Step 3. Evaluate the performance of estimation and store the calculation at each component
    sing_val_Chat=np.zeros(estim_R) #initilize the singular values of the estimated C
    C_est_err=np.zeros(estim_R) #initilize the cumulative estimation error
    predic_err_ontest=np.zeros(estim_R) #initilize the cumulative prediction error on test set
    predic_err_ontraining=np.zeros(estim_R) #initilize the cumulative prediction error on training set

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


