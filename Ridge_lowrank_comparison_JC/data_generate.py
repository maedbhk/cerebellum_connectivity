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

# Functions of generating simulated data for (1) ridge case, (2) low-rank case and (3) low-rank smoothness case.

def generate_X(n_obs, num_vox_cortex, smooth_level=2):
    """
    Generate X matrix with each row be the vectorized smoothness gaussian.
    """

    X=np.zeros((n_obs,num_vox_cortex))
    leng_sqr_cortex=int(math.sqrt(num_vox_cortex))
    #print(leng_sqr_cortex)

    for row in range(n_obs):
        #generate each row of X (smoothed)
        Z=np.random.normal(0,1,(leng_sqr_cortex,leng_sqr_cortex))
        cortex_mat = gaussian_filter(Z,smooth_level)
        #vectorize cortex_mat by rows
        X[row,:]=cortex_mat.reshape(-1)

    return X

#ridge case
#function generating the matrix C from N(0,sd=sigma)
def generate_C_ridge(nrow, ncol, sd):

    C=np.random.normal(loc=0.0, scale=sd, size=(nrow,ncol))

    return C

#low-rank smoothness case
#function generating the matrix C by first generating u's and v's, then implementing the low rank decomposition
def generate_C_lowranksmooth(rank, nrow, ncol, sd, smooth_level):
    #nrow: the number of cortex voxels
    #ncol: the number of cerebellum voxels
    leng_sqr_cortex=int(math.sqrt(nrow))
    leng_sqr_cere=int(math.sqrt(ncol))   

    C=np.zeros((nrow, ncol))
    #betas=np.zeros(rank*(nrow+ncol))
    U=np.zeros((nrow, rank))
    V=np.zeros((ncol, rank))

    for r in range(rank):
        #generating the spatial smoothed cortex vector u_r
        cortex_mat = gaussian_filter(np.random.normal(0,sd,(leng_sqr_cortex,leng_sqr_cortex)),smooth_level)
        #vectorize cortex_mat by rows
        cortex_u=cortex_mat.reshape(-1) 
        #betas[r*nrow:((r+1)*nrow)]=cortex_u
        U[:,r]=cortex_u

        #generating the spatial smoothed cerebellum vector v_r

        cere_mat = gaussian_filter(np.random.normal(0,sd,(leng_sqr_cere,leng_sqr_cere)),smooth_level)
        #vectorize cere_mat by rows
        cere_v=cere_mat.reshape(-1)      
        #betas[(rank*nrow+r*ncol):(rank*nrow+((r+1)*ncol))]=cere_v
        V[:,r]=cere_v

        C=C+np.outer(cortex_u,cere_v)

    return C, U, V

#low-rank case
#function generating the matrix C by first generating u's and v's from N(0, sigma), then implement the low-rank decomposition
def generate_C_lowrank(rank, nrow, ncol, sd): 

    C=np.zeros((nrow, ncol))
    #betas=np.zeros(rank*(nrow+ncol))
    U=np.random.normal(0,sd,(nrow,rank))
    V=np.random.normal(0,sd,(ncol,rank))

    C=U@V.T

    return C, U, V


from scipy.spatial import distance_matrix

def distance_mat(side_length):

    #side_length: the length of one side of the square

    x = np.arange(1,1+side_length, 1)

    xx, yy = np.meshgrid(x, x, sparse=False)

    #x_coor=pd.DataFrame(xx.reshape(-1,1))

    #y_coor=pd.DataFrame(yy.reshape(-1,1))

    coordi=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))

    return distance_matrix(coordi,coordi)