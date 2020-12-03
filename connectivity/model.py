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
    def to_dict(self): 
        d = {'coef_':self.coef_}
        return d


class L2regression(Ridge,ModelMixin): 
    """ 
        L2 regularized connectivity model
        simple wrapper for Ridge, implementing to_dict function
    """ 
    def __init__(self, alpha = 1):
        """ 
            Simply calls the superordinate construction - but does not fit intercept, as this removed by default in Dataset.get_data()
        """
        super().__init__(alpha=alpha,fit_intercept=False)

