import os
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

"""
connectivity models
A connectivity model is inherited from the sklearn class BaseEstimator
such that Ridge, Lasso, ElasticNet and other models can
be easily used. 

@authors: Maedbh King, Ladan Shahshahani, Jörn Diedrichsen 
"""
class ModelMixin:
    """ 
        This is a class that can give use extra behaviors or functions that we want our connectivity models to have - over an above the basic functionality provided by the stanard SK-learn BaseEstimator classes 
        As an example here is a function that serializes the fitted model
        Not used right now, but maybe potentially useful. Note that Mixin classes do not have Constructor! 
    """ 
    def to_dict(self): 
        d = {'coef_':self.coef_}
        return d

class L2regression(Ridge,ModelMixin): 
    """ 
        L2 regularized connectivity model
        simple wrapper for Ridge. It performs scaling by stdev, but not by mean before fitting and prediction
    """ 
    def __init__(self, alpha = 1):
        """ 
            Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        super().__init__(alpha=alpha,fit_intercept=False)

    def fit(self,X,Y): 
        Xs = X / np.sqrt(np.sum(X**2,0)/X.shape[0]) # Control scaling 
        return super().fit(Xs,Y)

    def predict(self,X):
        Xs = X / np.sqrt(np.sum(X**2,0)/X.shape[0]) # Control scaling 
        return super().predict(Xs)