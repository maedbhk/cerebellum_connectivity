import os
import numpy as np
import quadprog as qp
import cvxopt
import qpoases as oa
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
from sklearn.base import clone

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
        elif (self.solver=="qpoases"):
            #Setup data of first QP.
            ub  = np.ones((P1,))*100.0
            p = oa.PyQProblemB(P1)
            options = oa.PyOptions()
            options.enableFlippingBounds = oa.PyBooleanType.FALSE
            options.initialStatusBounds  = oa.PySubjectToStatus.INACTIVE
            options.numRefinementSteps   = 1
            p.setOptions(options)
            nWSR = np.array([2000])
            r1=p.init(G, -a[:,0], b, ub, nWSR)
            self.iter_ = np.ones((P2,1),dtype=int)*2000
            self.exit_ = np.zeros((P2),dtype=int)

            # Solve second QP.
            for i in range(P2):
                self.exit_[i]=p.hotstart(-a[:,i], b, ub, self.iter_[i,:])
                p.getPrimalSolution(self.coef_[i,:])

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T
