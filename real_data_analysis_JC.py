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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import timeit
import Class_SmoothLS_JC 

#Importing the data
# read the cortex data as X
#X_cortex02 = pd.read_csv("X_cortex02_sess_weight.csv").iloc[:, 1:] 
X_cortex02 = pd.read_csv("X_cortex02_full.csv").iloc[:, 1:] # read the large size data
X_cortex_eucldist02=pd.read_csv("X_cortex_eucldist02.csv").iloc[:, 1:] 
X_cortex_parceldist02=pd.read_csv("X_cortex_parceldist02.csv").iloc[:, 1:]

# read the cerebellum data as Y
#Y_cere02=pd.read_csv("Y_cere02_sess_weight.csv").iloc[:,1:]
Y_cere02=pd.read_csv("Y_cere02_suit3.csv").iloc[:,1:] # read the large size data
Y_cere_eucldist02=pd.read_csv("Y_cere_eucldist02.csv").iloc[:,1:]
Y_cere_parceldist02=pd.read_csv("Y_cere_parceldist02.csv").iloc[:,1:]

#Visualize the cortex dara
plt.imshow(X_cortex02)
plt.show()
X_cortex02.shape

X_cortex02.isnull().sum() # check the missing values 

XX_cortex=X_cortex02.fillna(0) # filling the missing values as 0

#Visualize the cerebellum data
plt.imshow(Y_cere02)
plt.show()
Y_cere02.shape

#First, preprocessing the data by centering the data X and Y
X_mean=np.array([XX_cortex.mean(axis=0)]*int(XX_cortex.shape[0]))
X_preprocess=XX_cortex-X_mean


Y_mean=np.array([Y_cere02.mean(axis=0)]*int(Y_cere02.shape[0]))
Y_preprocess=Y_cere02-Y_mean

# plot the centered X_cortex and Y_cere
fig, axe= plt.subplots(2,1)
axe[0].imshow(X_preprocess)
axe[1].imshow(Y_preprocess)
plt.show()


# Fitting the data
estim_R=50 #Cross validation for r=1:50
lamb=1
num_fold=10
time_used=np.zeros(num_fold)
pred_err_matr=np.zeros((estim_R,num_fold)) #This is a matrix used to stored results, with dimensions estim_R by num_fold
    
kf=KFold(n_splits=num_fold, shuffle=True)
suqr_err_test=np.zeros(num_fold)

fold_id=0
for train_ind, test_ind in kf.split(X_preprocess):
    print(f"Fold number: {fold_id+1}")
    X_train=X_preprocess.iloc[train_ind,:]
    #X_train.reset_index(inplace = True, drop = True)
    Y_train=Y_preprocess.iloc[train_ind,:]


    X_test=X_preprocess.iloc[test_ind,:]
    #X_test.reset_index(inplace = True, drop = True)
    Y_test=Y_preprocess.iloc[test_ind,:]

    start_time=timeit.default_timer()
    
    #create object
    Fit_real=Class_SmoothLS_JC.LowRankSmoothLS(rank=estim_R,lambda1=lamb)
    Fit_real.fit_iterative2(X_train, Y_train, X_cortex_eucldist02, Y_cere_eucldist02 ,X_dis_threshold=7.5, Y_dis_threshold=3)
    
    end_time=timeit.default.timer()
    
    #calculate the time for a fold 
    time_used[fold_id]=start_time-end_time
    print(f"Time used for fold {fold_id+1} is {time_used[fold_id]}.")
    
    #Loop in estim_R ranks to store the prediction error for each R
    for r_id in range(estim_R):
        C_est=Fit_real.U[:,0:(r_id+1)]@Fit_real.V[:,0:(r_id+1)].T
        pred_err_matr[r_id,fold_id]=LA.norm(Y_test-X_test@C_est)
    
    fold_id+=1

pred_err_matr=pd.DataFrame(pred_err_matr)
pred_err_matr.to_csv("pred_err_matrix.csv")

sum_pred_err=np.sum(pred_err_matr, axis=1)
sum_pred_err=pd.DataFrame(sum_pred_err)
sum_pred_err.to_csv("sum_pred_err.csv")
opt_rank=np.where(sum_pred_err == sum_pred_err.min())+1
print(f'The samllest pred err: {sum_pred_err[np.where(sum_pred_err == sum_pred_err.min())]}, with rank {opt_rank}.')
    
#plot(sum_squar_err) #Next, we need to find out the smallest sum of squared
                    #erroe to find optimal values of R
#Start from 2021 May 13 5:14 pm