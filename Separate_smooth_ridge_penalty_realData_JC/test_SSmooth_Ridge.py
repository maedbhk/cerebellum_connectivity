import numpy as np
import pandas as pd
#from scipy.optimize import minimize #this package is used when finding the minimizer of the objective
from numpy import linalg as LA #this package is used when calculating the norm
#import seaborn as sns #this package is used when ploting matrix C
#import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter

#from scipy.spatial import distance_matrix
import timeit
import datetime

import Separate_smooth_ridge_penalty_realData_JC.SSmooth_Ridge_class as SSR

def convert(n):
    return str(datetime.timedelta(seconds = n))

# First, read the real data

# read the cortex data as X
X_cortex02 = pd.read_csv("X_cortex0162_sc1_02_sess_weight.csv").iloc[:, 1:] 
X_cortex_eucldist02=pd.read_csv("X_cortex0162_eucldist_02.csv").iloc[:, 1:] 

# read the cerebellum data as Y
Y_cere02=pd.read_csv("Y_cere_sc1_02_sess_weight_suit3.csv").iloc[:,1:]
Y_cere_eucldist02=pd.read_csv("Y_cere_eucldist_02.csv").iloc[:,1:]

# Fitting the data

#First, prespecify parameters
estim_R=10
lamb_cor=0.07
lamb_cere=0.17
alp_cor=1.23
alp_cere=5.65

#start counting the time
start_time=timeit.default_timer()

Fit_real=SSR.SSmooth_Ridge_Opt(rank=estim_R,lambda_cor=lamb_cor, lambda_cere=lamb_cere, alpha_cor=alp_cor, alpha_cere=alp_cere)
Fit_real.fit(X_cortex02, Y_cere02, X_cortex_eucldist02, Y_cere_eucldist02 ,X_dis_threshold=17.5, Y_dis_threshold=3)

end_time=timeit.default_timer()
#Printing out the estimation time
print(f'The estimation time is: {convert(end_time-start_time)}.')

#Based on the model estimated, calculate the prediction error
X_cortex02_test = pd.read_csv("X_cortex0162_sc2_02_sess_weight.csv").iloc[:, 1:] 
Y_cere02_test=pd.read_csv("Y_cere_sc2_02_sess_weight.csv").iloc[:,1:]
Y_pred_test=Fit_real.predict(X_cortex02_test)

#Using L2 norm to evaluate the prediction error
Pred_err=LA.norm(Y_pred_test-Y_cere02_test)
print(f'The L2 norm prediction error is: {Pred_err}.')

for r in range(10):
    pred_res=np.array(Y_cere02)-np.array(X_cortex02)@Fit_real.U[:,0:(r+1)]@Fit_real.V[:,0:(r+1)].T
    pred_err=LA.norm(pred_res)
    print(f"rank={r+1}, pred_err={pred_err}")