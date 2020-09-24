import os
import numpy as np

from collections import defaultdict

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

"""
Created on Aug 10 10:04:24 2020
Models for running connectivity models
Any models can be added. Needs to follow a particular template
(see ExampleModel Class)

@authors: Maedbh King and Ladan Shahshahani
"""

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
        try: 
            model_fit = model.fit(self.X, self.Y)
        except:
            model_fit = model.fit(np.nan_to_num(self.X), self.Y)

        weights = model_fit.coef_

        return model_fit, weights
    
    def predict(self, model):
        try:
            pred = model.predict(self.X)
        except:
            pred = model.predict(np.nan_to_num(self.X))

        return pred

    def model_params(self, model):
        model_params = {'fit_intercept': model.fit_intercept,
                        'n_features': model.n_features_in_,
                        'normalize': model.normalize,
                        }
        return model_params
    
    def score(self, model):
        return model.score(self.X, self.Y)

class LinearModel(ModelUtils):

    def __init__(self, X, Y, config):
        """ does linear regression on 'X' and 'Y'
            Args: 
                X (array-like): shape (n_samples, n_features)
                Y (array-like): (n_samples,) or (n_samples, n_targets)
                config (dict): dictionary containing model training parameters
        """
        self.fit_intercept = False
        self.normalize = False
        self.X = X
        self.Y = Y
        self.config = config

    def define_model(self):
        return LinearRegression(fit_intercept = self.fit_intercept,
                                normalize = self.normalize)
    
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

    def __init__(self, X, Y, config):
        """ does l2 regression on 'X' and 'Y'
        parameters (i.e. lambdas) are given in `config`
            Args: 
                X (array-like): shape (n_samples, n_features)
                Y (array-like): (n_samples,) or (n_samples, n_targets)
                config (dict): dictionary containing model training parameters
        """
        self.fit_intercept = False
        self.normalize = False
        self.max_iter = None
        self.tol = 0.001
        self.random_state = None
        self.solver = 'auto'
        self.X = X
        self.Y = Y
        self.config = config
    
    def define_model(self):
        return Ridge(alpha = self.lam,
                    fit_intercept = self.fit_intercept,
                    normalize = self.normalize,
                    max_iter = self.max_iter,
                    tol = self.tol,
                    random_state = self.random_state,
                    solver = self.solver)
    
    def run(self):
        # check for model params (i.e. lambdass)
        if 'lambdas' in self.config:
            lambdas = self.config['lambdas']
        else:
            lambdas = [1]
        
        # loop over lambdas and get model data
        # initialize data lists
        data_dict = defaultdict(list)
        for self.lam in lambdas: 

            # define model
            model = self.define_model()

            # get model weights
            model_fit, weights = self.fit(model = model)

            # get model prediction
            Y_pred = self.predict(model=model_fit)

            # calculate R and R^2 scores
            # R2_py = self.score(model = model_fit)

            # get predictions
            R, R_vox = self.calculate_R(Y = self.Y, Y_pred = Y_pred)
            R2, R2_vox = self.calculate_R2(Y = self.Y, Y_pred = Y_pred)

            model_output = {'R': R, 'R_vox': R_vox, 'R2': R2, 'R2_vox': R2_vox,
                      'weights': weights, 'Y_pred': Y_pred}

            for k,v in model_output.items():
                data_dict[k].append(v)

        # get model params
        model_params = self.model_params(model = model)

        # update model params
        model_params.update({'model_params': 'lambdas',
                             'lambdas': lambdas,
                             'max_iter': self.max_iter,
                             'tol': self.tol,
                             'random_state': self.random_state,
                             'solver': self.solver,
                            })

        data_dict.update({'X_train': self.X, 'Y_train': self.Y, 'lambdas': lambdas})
        
        return model_params, data_dict

class ExampleModel(ModelUtils):
    
    def __init__(self, X, Y, config):
        """ does modelling (specifiy model name) on 'X' and 'Y'
        model-specific parameters (i.e. lambdas) are given in `config`
            Args: 
                X (array-like): shape (n_samples, n_features)
                Y (array-like): (n_samples,) or (n_samples, n_targets)
                config (dict): dictionary containing model training parameters
        """
        self.X = X
        self.Y = Y
        self.config = config
    
    def define_model(self):
        return None

    def run(self):
        """ run model on 'X' and 'Y'
            Returns: 
                model_params (dict): model parameters
                data_dict (dict): outputs of model training (e.g. 'weights', 'Y_pred')
        """
         
        model = self.define_model()

        model_params = None
        data_dict = None

        return model_params, data_dict

MODEL_MAP = {
    "linear_model": LinearModel,
    "l2_regress": L2Regress,
    }