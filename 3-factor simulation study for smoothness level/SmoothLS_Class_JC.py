import numpy as np
import pandas as pd

from numpy import linalg as LA #this package is used when calculating the norm


from scipy.ndimage import gaussian_filter
import random #set seed
import math
from scipy.spatial import distance_matrix
import statistics
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import timeit

class LowRankSmoothLS:
    """
        Lease squares estimation with low rank approximation 
        and smoothness constrains
    """
    
    def __init__(self, rank=2, lambda1=1):
        """
            Constructor
            Inputs:
                rank (integer): the rank of the coefficient matrix. This value decides how many ranks you will estimate
                                when you estimating the coefficient matrix C.
                lambda1 (integer): the tuning parameter for the smoothness penalty

        """
        self.rank = rank
        self.lambda1 = lambda1
    
  
    def fit_iterative2(self, X, Y, X_dist_mat, Y_dist_mat, X_dis_threshold=1, Y_dis_threshold=1):
        """
            Difference with function fit():use the new iterative method to estimate u and v saperatly and iterativly
            X: cortical matrix
            Y: cerebellumn matrix
            cor_dist_mat: cortical distance matrix
            cere_dist_mat: cerebellumn distance matrix
        """
        
        self.X=X
        self.Y=Y
        self.nrow=X.shape[1]
        self.ncol=Y.shape[1]
        self.L_X=Laplacian(X_dist_mat,X_dis_threshold) #obtain the graph Laplacian matrix for the cortex
        self.L_Y=Laplacian(Y_dist_mat,Y_dis_threshold) #obtain the graph Laplacian matrix for the cerebellum
        self.XTX=X.T@X
        self.XTY=X.T@Y
        self.len_betas=self.rank*(self.nrow+self.ncol)
        
        self.U=np.zeros((self.nrow, self.rank))
        self.V=np.zeros((self.ncol, self.rank))
        
        r=1
        Y_res=self.Y
        while (r<=self.rank):
            print(f'Estimating rank: {r}')
            u_r=np.random.uniform(low=-1, high=1, size=self.nrow)
            v_r=np.random.uniform(low=-1, high=1, size=self.ncol)
            #u_r=np.random.normal(0,1,size=self.nrow)
            #v_r=np.random.normal(0,1,size=self.ncol)
            dlt_u=LA.norm(u_r)
            dlt_v=LA.norm(v_r)
            C_0=np.outer(u_r, v_r)
            diff_C_norm=LA.norm(C_0)
            inner_id=1
            self.converge=True
            #Y_resi0=Y_res
            while (diff_C_norm>1e-4):
                print(f'Inner iteration number:{inner_id}')
                if (inner_id>2e6):
                    self.converge=False
                    print("convergence fail because the iteration number over 2e6.")
                    break #if the number of iteration is over 20,000 times, stop the iteration
                
                #when fix v, estimate u
                #solution_u=minimize(self.objective_u, self.u_r, method='BFGS', jac=True)
                #u_est, resid1, rank1, s1=LA.lstsq(LA.norm(v_r)**2*self.XTX+self.lambda1*self.L_X, self.X.T@Y_res@v_r,rcond=None)
                u_est=LA.solve(LA.norm(v_r)**2*self.XTX+self.lambda1*self.L_X, self.X.T@Y_res@v_r)
                print(f'u_est={u_est}')
                dlt_u=LA.norm(u_est-u_r)               
                norm_ur=LA.norm(u_est)
                print(f'||u_r||={norm_ur}')
                print(f'dlt_u={dlt_u}')
#                 if (norm_ur>1000):
#                     u_est=u_est/10
#                     norm_ur=LA.norm(u_est)
#                     print(f'new||u_r||={norm_ur}')
#                     print(f'New u_est={u_est}')
#                 if (norm_ur<1):
#                     u_est=u_est*10
#                     norm_ur=LA.norm(u_est)
#                     print(f'new||u_r||={norm_ur}')
#                     print(f'New u_est={u_est}')
                u_r=u_est
                
                #when fix u, estimate v
                #solution_v=minimize(self.objective_v, self.v_r, method='BFGS', jac=True)
                #v_est, resid2, rank2, s2=LA.lstsq(u_r@self.XTX@u_r*np.eye(self.ncol)+self.lambda1*self.L_Y, Y_res.T@self.X@u_r, rcond=None)
                v_est=LA.solve(u_r@self.XTX@u_r*np.eye(self.ncol)+self.lambda1*self.L_Y, Y_res.T@self.X@u_r)
                print(f'v_est={v_est}')
                dlt_v=LA.norm(v_est-v_r)                 
                norm_vr=LA.norm(v_est)
                print(f'||v_r||= {norm_vr}')
                print(f'dlt_v= {dlt_v}')
#                 if (norm_vr>1000):
#                     v_est=v_est/10
#                     norm_vr=LA.norm(v_est)
#                     print(f'new||v_r||={norm_vr}')
#                     print(f'New v_est={v_est}')
#                 if (norm_vr<1):
#                     v_est=v_est*10
#                     norm_vr=LA.norm(v_est)
#                     print(f'new||v_r||={norm_vr}')
#                     print(f'New v_est={v_est}')
                v_r=v_est
                
                C_r=np.outer(u_r, v_r)
                norm_Cr=LA.norm(C_r)
                diff_C_norm=LA.norm(C_r-C_0)
                print(f"||C(i)-C(i-1)||={diff_C_norm}")
                print(f'||C_(i)||={norm_Cr}')
                C_0=C_r
                #Y_resi=np.array(Y_res)-self.X@C_r
                #norm_Y_resi=LA.norm(Y_resi)
                #print(f'||Y_reis_r_{inner_id}||={norm_Y_resi} ')
                #norm_diff_Y_resi=LA.norm(Y_resi-Y_resi0)
                #print(f'norm_Y_resi{i+1}-norm_Y_resi{i}={norm_diff_Y_resi}')
                #Y_resi0=Y_resi
                inner_id+=1
                #end inner while loop

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
            r+=1
            #end outer while loop
        
        self.betas=np.concatenate((self.U.T.reshape(-1), self.V.T.reshape(-1)), axis=None)  
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