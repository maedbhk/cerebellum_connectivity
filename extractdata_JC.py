#question: what is the difference between data.get_diatance_matrix() and data.eucl_distance()?

#First, import h5py if not working, uninstall first, then install h5py
import h5py
#import essential packages (same as plot_taskmaps.ipynb)
from connectivity.tasks import get_betas, plot_task_maps_cerebellum
import SUITPy as suit
import surfAnalysisPy as surf
import numpy as np
import connectivity.data as data
import connectivity.constants as const
import os
import matplotlib.pyplot as plt
import pandas as pd



#extract the cortex data X
Xdata=data.Dataset(experiment="sc1",glm="glm7",roi="tessels1442",subj_id="s02")
Xdata.load_mat()
X, X_info=Xdata.get_data(averaging="none", weighting=False) #averaging="none": the largest one; obtain the original data witout any averaging
                                                            #averaging="exp", the smallest one
                                                            #averaging ="sess" the middle size one

#extrct the cerebellum data Y
Ydata=data.Dataset(experiment="sc1",glm="glm7",roi="cerebellum_suit",subj_id="s02")
Ydata.load_mat() # Import the data from Matlab
Y, Y_info = Ydata.get_data(averaging="none",weighting = False)

#obtain the distance matrix for cortex
Xdist, Xcoord =data.get_distance_matrix('tessels1442')

#obtain the distance matrix for cerebellum
Ydist, Ycoord =data.get_distance_matrix('cerebellum_suit')


X_cortex=pd.DataFrame(X)       
Y_cere=pd.DataFrame(Y) 
X_cortex_dist=pd.DataFrame(Xdist)
Y_cere_dist=pd.DataFrame(Ydist)   
X_cortex_coord=pd.DataFrame(Xcoord)   
Y_cere_coord=pd.DataFrame(Ycoord)  
X_cortex.to_csv('X_cortex_sc1_02_none_weightF.csv')
Y_cere.to_csv('Y_cere_sc1_02_none_weightF.csv')
X_cortex_dist.to_csv('X_cortex_parceldist02.csv')
Y_cere_dist.to_csv('Y_cere_parceldist02.csv')

X_cortex_eucldist=pd.DataFrame(data.eucl_distance(Xcoord))
Y_cere_eucldist=pd.DataFrame(data.eucl_distance(Ycoord))
X_cortex_eucldist.to_csv("X_cortex_eucldist02.csv")
Y_cere_eucldist.to_csv("Y_cere_eucldist02.csv")