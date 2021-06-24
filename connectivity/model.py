from operator import index
import os
import numpy as np
import pandas as pd
import quadprog as qp
# import cvxopt
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.base import clone
from scipy import stats
import connectivity.evaluation as ev # will be used in stepwise regression


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
        data = {"coef_": self.coef_}
        return data


class L2regression(Ridge, ModelMixin):
    """
    L2 regularized connectivity model
    simple wrapper for Ridge. It performs scaling by stdev, but not by mean before fitting and prediction
    """

    def __init__(self, alpha=1):
        """
        Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        super().__init__(alpha=alpha, fit_intercept=False)

    def fit(self, X, Y):
        self.scale_ = np.sqrt(np.nansum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return super().fit(Xs, Y)

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)

class LASSO(Lasso, ModelMixin):
    """
    L2 regularized connectivity model
    simple wrapper for Ridge. It performs scaling by stdev, but not by mean before fitting and prediction
    """

    def __init__(self, alpha=1):
        """
        Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        super().__init__(alpha=alpha, fit_intercept=False)

    def fit(self, X, Y):
        self.scale_ = np.sqrt(np.nansum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return super().fit(Xs, Y)

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)

class WTA(LinearRegression, ModelMixin):
    """
    WTA model
    It performs scaling by stdev, but not by mean before fitting and prediction
    """

    def __init__(self, positive=False):
        """
        Simply calls the superordinate construction - but does not fit intercept, as this is tightly controlled in Dataset.get_data()
        """
        super().__init__(positive=positive, fit_intercept=False)

    def fit(self, X, Y):
        self.scale_ = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        super().fit(Xs, Y)
        self.labels = np.argmax(self.coef_, axis=1)
        wta_coef_ = np.amax(self.coef_, axis=1)
        self.coef_ = np.zeros((self.coef_.shape))
        num_vox = self.coef_.shape[0]
        # for v in range(num_vox):
        #     self.coef_[v, self.labels[v]] = wta_coef_[v]
        self.coef_[np.arange(num_vox), self.labels] = wta_coef_
        return self.coef_, self.labels

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)

class WNTA(Ridge, ModelMixin):
    """
    from sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
    This Sequential Feature Selector adds (forward selection) or removes (backward selection) features 
    to form a feature subset in a greedy fashion. At each stage, this estimator chooses the best feature
    to add or remove based on the cross-validation score of the estimator (here the estimator is a linear regression class).

    can be used to fit winner-take-all (set n = 1 (default)) and winner-n-take-all
    """

    def __init__(self, alpha, positive = False, n = 1):
        """
        defines a forward sequential feature selector
        """
        self.n = n
        # defining a feature selector object (with forward selection)
        self.selector = SequentialFeatureSelector(LinearRegression(fit_intercept=False), n_features_to_select=self.n)
        super(Ridge, self).__init__(alpha=alpha, fit_intercept=False)
        self.positive = positive
        
    def fit(self, X, Y):
        self.scale_ = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        ## fit and return the selected features (cortical parcels)
        Xs = self.selector.fit_transform(Xs, Y)
        ## do a ridge regression with the selected cortical tessels
        super(Ridge, self).fit(Xs, Y)
        return self.coef_

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        # just get the selected features
        Xs = self.selector.transform(Xs)
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)

class WNTA2(LinearRegression, Ridge, ModelMixin):

    """
    from sklearn documentation:
    Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
    """

    def __init__(self, alpha, positive = False, n = 1):
        """
        defines a forward sequential feature selector
        """
        self.n = n
        # defining a feature selector object (with forward selection)
        
        super(Ridge, self).__init__(alpha=alpha, fit_intercept=False)
        self.positive = positive
        
    def fit(self, X, Y):

        self.scale_ = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        ## recursive feature selection with cross validation.
        ### returns the selected features that resulted in the highest cv score
        self.selector = RFECV(LinearRegression(fit_intercept=False))
        self.selector.fit(Xs, Y)

        # attributes of the selector:
        # ranking_ : returns ranking of the features
        # support_ : returns True for selected features, False otherwise

        Xs = self.selector.transform(Xs)   # getting the selected features
        self.n_selected_feat = self.selector.n_features_ # how many features were selected
        self.selected_feat   = range(X.shape)[self.selector.support_] # what are the selected features
        ## do a ridge regression with the selected cortical tessels
        super(Ridge, self).fit(Xs, Y)
        return self.coef_

    def _predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        # just get the selected features
        Xs = self.selector.transform(Xs)
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)

class NNLS(BaseEstimator, ModelMixin):
    """
    Fast implementation of a multivariate Non-negative least squares (NNLS) regression
    Allows for both L2 and L1 penality on regression coefficients (i.e. Elastic-net like).
    Regression model is transformed into a quadratic programming problem and then solved
    using the  quadprog module
    """

    def __init__(self, alpha=0, gamma=0, solver = "cvxopt"):
        """
        Constructor. Input:
            alpha (double):
                L2-regularisation
            gamma (double):
                L1-regularisation (0 def)
            solver
                Library for solving quadratic programming problem
        """
        self.alpha = alpha
        self.gamma = gamma
        self.solver = solver

    def fit(self, X, Y):
        """
        Fitting of NNLS model including scaling of X matrix
        """
        N, P1 = X.shape
        P2 = Y.shape[1]
        self.scale_ = np.sqrt(np.sum(X ** 2, 0) / X.shape[0])
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        G = Xs.T @ Xs + np.eye(P1) * self.alpha
        a = Xs.T @ Y - self.gamma
        C = np.eye(P1)
        b = np.zeros((P1,))
        self.coef_ = np.zeros((P2, P1))
        if (self.solver=="quadprog"):
            for i in range(P2):
                self.coef_[i, :] = qp.solve_qp(G, a[:, i], C, b, 0)[0]
        elif (self.solver=="cvxopt"):
            Gc = cvxopt.matrix(G)
            Cc = cvxopt.matrix(-1*C)
            bc = cvxopt.matrix(b)
            inVa = cvxopt.matrix(np.zeros((P1,)))
            for i in range(P2):
                ac = cvxopt.matrix(-a[:,i])
                sol = cvxopt.solvers.qp(Gc,ac,Cc,bc,initvals=inVa)
                self.coef_[i, :] = np.array(sol['x']).reshape((P1,))
                inVa = sol['x']


    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T 

class PLSRegress(PLSRegression, ModelMixin):
    """
        PLS regression connectivity model
        for more info:
            https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.pls.PLSRegression.html
            from sklearn.pls import PLSCanonical, PLSRegression, CCA
            https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
            pls2_mod = PLSRegression(n_components = N, algorithm = method)
    """

    def __init__(self, n_components = 1):
        super().__init__(n_components =n_components)
        
    def fit(self, X, Y):
        """
        uses nipals algorithm 
        """

        Xs = X / np.sqrt(np.sum(X**2,0)/X.shape[0]) # Control scaling 

        Xs = np.nan_to_num(Xs)
        return super().fit(Xs,Y)

    def predict(self, X):
        Xs = X / np.sqrt(np.sum(X**2,0)/X.shape[0]) # Control scaling 
        Xs = np.nan_to_num(Xs)
        return super().predict(Xs)