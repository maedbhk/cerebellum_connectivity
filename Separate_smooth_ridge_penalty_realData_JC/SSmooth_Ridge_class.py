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
    
class SSmooth_Ridge:
    """
        Low-rank least squares estimation:
        Separate smoothness penaltyies on the cortex and cerebellum
        Separate ridge penaltyies on the cortex and cerebellum
    """
    
    def __init__(self, rank=5, lambda_cor=1, lambda_cere=1, alpha_cor=1, alpha_cere=1):
        """
            Constructor
            Inputs:
                rank (integer): the rank of the coefficient matrix. This value decides how many ranks you will estimate
                                when you estimating the coefficient matrix C.
                lambda_cor: the tuning parameter for the smoothness penalty on the cortex
                lambda_cere: the tuning parameter for the smoothness penalty on the cerebellum
                alpha_cor: the tuning parameter for the ridge on cortex
                alpha_cere: the tuning parameter for the ridge on cerebellum
        """
        self.rank = rank
        self.lambda_cor= lambda_cor
        self.lambda_cere=lambda_cere
        self.alpha_cor=alpha_cor
        self.alpha_cere=alpha_cere
    
   
    def fit_ite_comp(self, X, Y, X_dist_mat, Y_dist_mat, X_dis_threshold=17.5, Y_dis_threshold=3):
        """
            Difference with function fit():use the new iterative method to estimate u and v saperatly and iterativly
            X: cortical matrix
            Y: cerebellumn matrix
            cor_dist_mat: cortical distance matrix
            cere_dist_mat: cerebellumn distance matrix
            X_dis_threshold: the distance threshold to decide the connection between cortex voxels
            Y_dis_threshold: the distance threshold to decide the connection between cerebellum voxels
        """
        
        self.X=X
        self.Y=Y
        self.C_nrow=X.shape[1]
        self.C_ncol=Y.shape[1]
        self.L_X=Laplacian(X_dist_mat,X_dis_threshold) #obtain the graph Laplacian matrix for the cortex
        self.L_Y=Laplacian(Y_dist_mat,Y_dis_threshold) #obtain the graph Laplacian matrix for the cerebellum
        self.L_X_ridge=self.L_X+self.alpha_cor*np.eye(self.C_nrow)
        self.L_Y_ridge=self.L_Y+self.alpha_cere*np.eye(self.C_ncol)
        self.XTX=X.T@X
        self.XTY=X.T@Y
        
        self.U=np.zeros((self.C_nrow, self.rank))
        self.V=np.zeros((self.C_ncol, self.rank))
        
        r=1
        Y_res=self.Y
        while (r<=self.rank):
            #start counting the time
            start_time=timeit.default_timer()
            
            print(f'Estimating rank: {r}')
            u_r=np.random.uniform(low=-1, high=1, size=self.C_nrow)
            v_r=np.random.uniform(low=-1, high=1, size=self.C_ncol)
            #u_r=np.random.normal(0,1,size=self.nrow)
            #v_r=np.random.normal(0,1,size=self.ncol)
            dlt_u=1
            dlt_v=1
            C_0=np.outer(u_r, v_r)
            diff_C_norm=1
            inner_id=1
            self.converge=True

            while (diff_C_norm>1e-3):
                print(f'Inner iteration number:{inner_id}')
                if (inner_id>2e6):
                    self.converge=False
                    print("convergence fail because the iteration number over 2e6.")
                    break #if the number of iteration is over 20,000 times, stop the iteration
                
                #when fix v, estimate u
                #solution_u=minimize(self.objective_u, self.u_r, method='BFGS', jac=True)
                #u_est, resid1, rank1, s1=LA.lstsq(LA.norm(v_r)**2*self.XTX+self.lambda1*self.L_X, self.X.T@Y_res@v_r,rcond=None)
                start_time1=timeit.default_timer()
                u_est=LA.solve(LA.norm(v_r)**2*self.XTX+self.lambda_cor*self.L_X_ridge, self.X.T@Y_res@v_r)
                end_time1=timeit.default_timer()
                dlt_time1=convert(end_time1-start_time1)
                print(f"time used to calulate u_est: {dlt_time1}")
                print(f'u_est={u_est[:10]}')
                dlt_u=LA.norm(u_est-u_r)               
                norm_ur=LA.norm(u_est)
                print(f'||u_r||={norm_ur}')
                print(f'dlt_u={dlt_u}')
                u_r=u_est
                
                #when fix u, estimate v
                #solution_v=minimize(self.objective_v, self.v_r, method='BFGS', jac=True)
                #v_est, resid2, rank2, s2=LA.lstsq(u_r@self.XTX@u_r*np.eye(self.ncol)+self.lambda1*self.L_Y, Y_res.T@self.X@u_r, rcond=None)
                start_time2=timeit.default_timer()
                v_est=LA.solve(u_r.T@self.XTX@u_r*np.eye(self.C_ncol)+self.lambda_cere*self.L_Y_ridge, Y_res.T@self.X@u_r)
                end_time2=timeit.default_timer()
                dlt_time2=convert(end_time2-start_time2)
                print(f"time used to calulate v_est: {dlt_time2}")
                print(f'v_est={v_est[:10]}')
                dlt_v=LA.norm(v_est-v_r)                 
                norm_vr=LA.norm(v_est)
                print(f'||v_r||= {norm_vr}')
                print(f'dlt_v= {dlt_v}')
                v_r=v_est
                
                C_r=np.outer(u_r, v_r)
                norm_Cr=LA.norm(C_r)
                diff_C_norm=LA.norm(C_r-C_0)
                print(f"||C(i)-C(i-1)||={diff_C_norm}")
                print(f'||C_(i)||={norm_Cr}')
                C_0=C_r

                inner_id+=1
                #end inner while loop
                #stop counting time
            end_time=timeit.default_timer()
            time_used=convert(end_time-start_time)
            print(f"Time used for rank {r}: {time_used}")

            if (self.converge==False):
                print(f"inner iteration number >2e6, thus stop iteration to estimate the u_r and v_r. All singular values and vectors are zeros from rank {r}.")
                break # if at rth component, the result is not converge, then from this component, all singular vales,
                      # singular vectors are all zeros
            #print(f'Rank {r} estimation finishes. ||C_r||={norm_Cr}')
            self.U[:,(r-1)]=u_r
            self.V[:,(r-1)]=v_r
            Y_res=np.array(Y_res)-self.X@np.outer(u_r, v_r)
            norm_Y_res=LA.norm(Y_res)
            print(f'norm of Y_res: {norm_Y_res}')
            #end outer while loop
            r+=1

        self.C_est=self.U@self.V.T
        return self
    
def Laplacian(dist_mat, dis_threshold=1):
    """
        Obtain the Laplacian matrix based on the distance matrix
        dist_mat: distance matrix, based on which the Laplacian matrix is obtained
    """
    adjacen_mat=np.where(dist_mat <= dis_threshold, 1, 0)-np.eye(dist_mat.shape[1]) #obtain the adjacenct matrix 
    num_edge=np.sum(adjacen_mat, axis=0) #sum over rows to get the number of closed voxels 
    L=np.diag(num_edge)-adjacen_mat
        
    return L

def convert(n):
    return str(datetime.timedelta(seconds = n))