import sys
import glob
import numpy as np
# import json
# import deepdish as dd
import pandas as pd
# import connectivity.io as cio
# from connectivity.data import Dataset
# import connectivity.constants as const
import matplotlib.pyplot as plt
import connectivity.model as model # import models
import connectivity.evaluation as ev # import evaluation methods
from numpy import linalg as LA #this package is used when calculating the norm

#First, read the data
# read the cortex data as X
#X_cortex02 = pd.read_csv("X_cortex02_sess_weight.csv").iloc[:, 1:] 
X_cortex02 = pd.read_csv("X_cortex_sc1_02_none_weightF.csv").iloc[:, 1:] # read the large size data
plt.imshow(X_cortex02)
plt.show()
XX_cortex=X_cortex02.fillna(0) # filling the missing values as 0
#X_cortex_eucldist02=pd.read_csv("X_cortex_eucldist02.csv").iloc[:, 1:] 
#X_cortex_parceldist02=pd.read_csv("X_cortex_parceldist02.csv").iloc[:, 1:]

# read the cerebellum data as Y
#Y_cere02=pd.read_csv("Y_cere02_sess_weight.csv").iloc[:,1:]
Y_cere02=pd.read_csv("Y_cere_sc1_02_none_weightF.csv").iloc[:,1:] # read the large size data
#Y_cere_eucldist02=pd.read_csv("Y_cere_eucldist02.csv").iloc[:,1:]
#Y_cere_parceldist02=pd.read_csv("Y_cere_parceldist02.csv").iloc[:,1:]

#Using ridge model to fit the data
ridge_model=model.L2regression(alpha=1) #using the default value alpha=1
ridge_model.fit(XX_cortex, Y_cere02)

#predicting on the same training dataset
Y_pred=ridge_model.predict(XX_cortex)
pred_err_training=LA.norm(Y_pred-Y_cere02)
#5.637440436010949

#predicting on the different testset
X_cortex_sc2_02 = pd.read_csv("X_cortex_sc2_02_none_weightF.csv").iloc[:, 1:] #already fill the missing values as 0
plt.imshow(X_cortex_sc2_02)
plt.show()
Y_cere_sc2_02=pd.read_csv("Y_cere_sc2_02_none_weightF.csv").iloc[:,1:]

#select the first 735 observations
X_test=X_cortex_sc2_02.iloc[:736,:]
Y_test=Y_cere_sc2_02.iloc[:736,:]

#test the prediction rate on the testset
Y_pred_test=ridge_model.predict(X_test)
pred_err_test=LA.norm(Y_pred_test-Y_test)
pred_err_test**2
#1043.9094313609005

R, R_voxel=ev.calculate_R(Y_test, Y_pred_test) #R=0.30987913177390647; R_voxl=...
plt.plot(R_voxel)
plt.show()

R2, R2_voxel=ev.calculate_R2(Y_test, Y_pred_test)
plt.plot(R2_voxel)
plt.show()
# R2: -0.2004265511846226

