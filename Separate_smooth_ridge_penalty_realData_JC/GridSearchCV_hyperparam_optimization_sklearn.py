
# 
from scipy.stats import loguniform
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

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


# define model
model=SSR.SSmooth_Ridge_Opt()
#SSmooth_Ridge has five arguments: rank=5, lambda_cor=1, lambda_cere=1, alpha_cor=1, alpha_cere=1

# define search space
space = dict()
# space['lambda_cor'] = [1e-5,  1e-3,  1e-1, 1, 100,  10000, 1000000]
# space['lambda_cere'] = [1e-5,  1e-3,  1e-1, 1, 100,  10000, 1000000]
# space['alpha_cor'] = [1e-5,  1e-3,  1e-1, 1, 100,  10000, 1000000]
# space['alpha_cere'] = [1e-5,  1e-3,  1e-1, 1, 100,  10000, 1000000]
space['rank']=[1]
space['lambda_cor'] = [ 0.1, 1,  100, 1000]
space['lambda_cere'] = [ 0.1,1, 100, 1000]
space['alpha_cor'] = [ 0.1, 1, 10, 100, 1000]
space['alpha_cere'] = [ 0.1, 1, 10, 100, 1000]

# define evaluation
cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1)

# define search
search = GridSearchCV(model, space, scoring='neg_mean_squared_error', n_jobs=2, cv=cv)

#Start counting time
start_time=timeit.default_timer()
# execute search
result = search.fit(X_cortex02, Y_cere02, X_dist_mat=X_cortex_eucldist02, Y_dist_mat=Y_cere_eucldist02)
#End counting time
end_time=timeit.default_timer()

time_used=convert(end_time-start_time)
print(f"TIme used to find the optimal hyperparameters is:{time_used}")


# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
print(f'The cv results:{result.cv_results_}') #A dict with keys as column headers and values as columns, that can be
                                            #imported into a pandas ``DataFrame``.

print(f"The best estimator: {result.best_estimator_}") #Estimator that was chosen by the search, i.e. estimator 
                #which gave highest score (or smallest loss if specified)
                #on the left out data. Not available if ``refit=False``.
print(f"The best index: {result.best_index_}") #The index (of the ``cv_results_`` arrays) which corresponds to the best candidate parameter setting.
print(f"Score function: {result.scorer_}") #Scorer function used on the held out data to choose the best parameters for the model.
print(f"Number of split in cross validation: {result.n_splits_}")
print(f"Time used for refit the final model: {convert(result.refit_time_)}")
pd.DataFrame(result.cv_results_).to_csv("cv_results_grid_rank1.csv")
pd.DataFrame(result.cv_results_)