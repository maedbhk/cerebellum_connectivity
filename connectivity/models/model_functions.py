import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

class ModelUtils:
    """ general purpose utils for modelling connectivity data
    """

    def __init__(self):
        pass

    def calculate_R(self, Y, Y_pred):
        # Calculating R 
        res = Y - Y_pred

        SYP = np.nansum(Y*Y_pred, axis = 0);
        SPP = np.nansum(Y_pred*Y_pred, axis = 0);
        SST = np.sum((Y - Y.mean()) ** 2, axis = 0) # use np.nanmean(Y) here?

        R = np.nansum(SYP)/np.sqrt(np.nansum(SST)*np.nansum(SPP));
        R_vox = SYP/np.sqrt(SST*SPP) # per voxel

        return R, R_vox

    def calculate_R2(self, Y, Y_pred):
        # Calculating R2
        res = Y - Y_pred

        SSR = np.nansum(res **2, axis = 0) # remember: without setting the axis, it just "flats" out the whole array and sum over all
        SST = np.sum((Y - Y.mean()) ** 2, axis = 0) # use np.nanmean(Y) here??

        R2 = 1 - (np.nansum(SSR)/np.nansum(SST))
        R2_vox = 1 - (SSR/SST)

        return R2, R2_vox

    def fit(self, model):
        model_fit = model.fit(self.X, self.Y)
        weights = model_fit.coef_

        return model_fit, weights
    
    def predict(self, model):
        return model.predict(self.X)

    def model_params(self, model):
        model_params = {'fit_intercept': model.fit_intercept,
                        'n_features': model.n_features_in_,
                        'normalize': model.normalize,
                        }
        return model_params
    
    def score(self, model):
        return model.score(self.X, self.Y)

class LinearModel(ModelUtils):

    def __init__(self, X, Y):
        self.fit_intercept = False
        self.normalize = False
        self.X = X
        self.Y = Y

    def define_model(self):
        return LinearRegression(fit_intercept = self.fit_intercept, normalize = self.no)
    
    def run(self):
        # define model
        model = self.define_model()

        # get model weights
        model_fit, weights = self.fit(model = model)

        # get model params
        model_params = self.model_params(model = model)

        # get model prediction
        Y_pred = self.predict(model = model_fit)

        # calculate R and R^2 scores
        # R2_py = self.score(model = model_fit)

        R, R_vox = self.calculate_R(Y = self.Y, Y_pred = Y_pred)
        R2, R2_vox = self.calculate_R2(Y = self.Y, Y_pred = Y_pred)

        data_dict = {'X_train': self.X, 'Y_train': self.Y,
                    'weights': weights, 'Y_pred': Y_pred,
                    'R': R, 'R2': R2, 'R_vox': R_vox,
                    'R2_vox': R2_vox}
        
        return model_params, data_dict

class L2Regress(ModelUtils):

    def __init__(self, X, Y):
        self.fit_intercept = False
        self.normalize = False
        self.max_iter = None
        self.tol = 0.001
        self.random_state = None
        self.solver = 'auto'
        self.X = X
        self.Y = Y
    
    def define_model(self):
        return Ridge(alpha = self.lam,
                    fit_intercept = self.fit_intercept,
                    normalize = self.normalize,
                    max_iter = self.max_iter,
                    tol = self.tol,
                    random_state = self.random_state,
                    solver = self.solver)
    
    def run(self, **kwargs):
        if kwargs.get('lambdas'):
            lambdas = kwargs['lambdas']
        else:
            lambdas = [1] # default is 1
        
        # loop over lambdas and get model data
        # initialize data lists
        data_dict = {'weights': [], 'Y_pred': [], 'R': [], 'R_vox': [], 'R2': [], 'R2_vox': []}
        for self.lam in lambdas: 

            # define model
            model = self.define_model()

            # get model weights
            model_fit, weights = self.fit(model = model)

            # get model prediction
            Y_pred = self.predict(model = model_fit)

            # calculate R and R^2 scores
            # R2_py = self.score(model = model_fit)

            R, R_vox = self.calculate_R(Y = self.Y, Y_pred = Y_pred)
            R2, R2_vox = self.calculate_R2(Y = self.Y, Y_pred = Y_pred)

            # append to dict
            data_dict['weights'].append(weights)
            data_dict['Y_pred'].append(Y_pred)
            data_dict['R'].append(R)
            data_dict['R_vox'].append(R_vox)
            data_dict['R2'].append(R_vox)
            data_dict['R2_vox'].append(R_vox)

        # get model params
        model_params = self.model_params(model = model)

        # update model params
        model_params.update({'alphas': lambdas,
                             'max_iter': self.max_iter,
                             'tol': self.tol,
                             'random_state': self.random_state,
                             'solver': self.solver,
                            })

        data_dict.update({'X_train': self.X, 'Y_train': self.Y, 'lambdas': lambdas})
        
        return model_params, data_dict

MODEL_MAP = {
    "linear_model": LinearModel,
    "l2_regress": L2Regress,
    }