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

class LinearModel(ModelUtils):

    def __init__(self, X, Y):
        self.fit_intercept = False
        self.X = X
        self.Y = Y

    def define_model(self):
        return LinearRegression(fit_intercept = self.fit_intercept)
    
    def fit(self, model):
        model_fit = model.fit(self.X, self.Y)
        weights = model_fit.coef_

        return model_fit, weights
    
    def predict(self, model):
        return model.predict(self.X)

    def score(self, model):
        return model.score(self.X, self.Y)

    def model_params(self, model):
        model_params = {'fit_intercept': model.fit_intercept,
                        'n_features': model.n_features_in_,
                        'normalize': model.normalize,
                        }
        return model_params
    
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
        self.X = X
        self.Y = Y
    
    def fit(self, X, Y):
        # M   = {}
        lam = kwargs['args']
        model = Ridge(alpha = lam, fit_intercept = self.fit_intercept)
        fitted_model = model.fit(self.X, self.Y)
        # M['lambda']  = lam
        # M['W']       = fitted_model.coef_
        weights = fitted_model.coef_

        Y_pred = fitted_model.predict(X)

        return weights, Y_pred

MODEL_MAP = {
    "linear_model": LinearModel,
    "l2_regress": L2Regress,
    }