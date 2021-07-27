import numpy as np
import pandas as pd
#from scipy.optimize import minimize #this package is used when finding the minimizer of the objective
from numpy import linalg as LA #this package is used when calculating the norm
#import seaborn as sns #this package is used when ploting matrix C
#import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter
import random #set seed
import math
#from scipy.spatial import distance_matrix
import timeit
import datetime
import scipy
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
lamb_cor=1
lamb_cere=1
alp_cor=1
alp_cere=1

#start counting the time
start_time=timeit.default_timer()

Fit_real=SSR.SSmooth_Ridge(rank=estim_R,lambda_cor=lamb_cor, lambda_cere=lamb_cere, alpha_cor=alp_cor, alpha_cere=alp_cere)
Fit_real.fit_ite_comp_sparse(X_cortex02, Y_cere02, X_cortex_eucldist02, Y_cere_eucldist02 ,X_dis_threshold=17.5, Y_dis_threshold=3)

end_time=timeit.default_timer()
    
#calculate the time 
time_used=convert(end_time-start_time)
time_used