from operator import index
import os
import time
import numpy as np
import pandas as pd
# import quadprog as qp
# import cvxopt
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
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
        self.labels = self.labels + 1 # we don't want zero-indexed label
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

class WINNERS(ModelMixin):

    def __init__(self, n_features_to_select = 1):
        self.n_featrues_to_select = n_features_to_select
        
    def add_features(self, X, y, selected = []):
       
        """
        1. start with evaluation of individual features
        2. select the one feature that results in the best performance
            ** what is the best? That depends on the selected evaluation criteria (in this case it can be R)
        3. Consider all the possible combinations of the selected feature and another feature and select the best combination
        4. Repeat 1 to 3 untill you have the desired number of features

        Args: 
        X(np.ndarray)   -    design matrix   
        Y(np.ndarray)   -    response variables
        n(int)          -    number of features to select
        """
        remaining = list(set(range(X.shape[1])) - set(selected)) #list containing features that are to be examined

        # 2. loop over features
        while (remaining) and (len(selected) < self.n_featrues_to_select): # while remaining is not empty and n features are not selected 
            
            scores = pd.Series(np.empty((len(remaining))), index=remaining) # the scores will be stored in this 
            for i in remaining:
        
                candidates = selected +[i] # list containing the current features that will be used in regression
                # fit the model
                ## get the features from X
                X_feat = X[:, list(candidates)]

                ## scale X_feat
                scale_ = np.sqrt(np.nansum(X_feat ** 2, 0) / X_feat.shape[0])
                Xs = X_feat / scale_
                Xs = np.nan_to_num(Xs) # there are 0 values after scaling

                ## fit the model
                ### 1. get regression coefficient
                B = (np.linalg.inv(X_feat.T@X_feat))@X_feat.T@y
                ### 2. get predicted values
                y_pred = X_feat@B
                # R = y - X_feat@B
                # mod = LinearRegression(fit_intercept=False).fit(Xs, y)

                ## get the score and put it in scores
                ## get the score
                score_i, _    = ev.calculate_R(y, y_pred)
                # print(score_i)
                scores.loc[i] = score_i
                                

            # find the feature/feature combination with the best score
            best       = scores.idxmax()
            selected.append(best)

            # update remaining
            ## remove the selected feature/features from remaining
            remaining.remove(best)

        return selected

    def set_support_(self, X, Y, support_ = None):
        """
        gets the support (and updates it) for all the voxels
        support_ can then be used to get the selected features
        Ars:
            X(ndarray)          : contains regressors (cortical regions)
            Y(ndarray)          : contains responses (cerebellar voxels)
            support_(ndarray)   : initial mask to select features. None: starts from scratch with all zeros
        """

        if support_ is None:
            # starting from scratch
            self.support_= np.zeros((Y.shape[1], X.shape[1]))
        else:
            self.support_ = support_

        # loop over voxels
        for vox in range(Y.shape[1]):
            # print(vox, end = "", flush=True)

            if np.any(Y[:, vox]):
                # get the selected features for the current voxel
                initial_feats = list(np.where(self.support_[vox, :] == 1)[0])

                # add features to the selected set
                feats = self.add_features(X, Y[:, vox], selected = initial_feats)

                # update support 
                self.support_[vox, feats] = int(1)

        return self.support_
        
class WNTA(Ridge, ModelMixin):

    def __init__(self, winner_model = None, alpha = 0, n_features_to_select = 1):
        """
        should be initialized with an instance of WINNERS class. 
        if None is entered, it will start from scratch, create an instance of WINNERS 
        and get the support_ for selecting features. Otherwise, It uses the support_ attribute
        of the WINNERS class
        """

        super(Ridge, self).__init__(fit_intercept=False, alpha = alpha)

        if winner_model is None:
            # initialize a winner model class
            self.winner_model = WINNERS(n_features_to_select = n_features_to_select)
        else: 
            self.winner_model = winner_model
            

        self.n_features_to_select = n_features_to_select
          
    def fit(self, X, Y):

        # get the scaling
        self.scale_ = np.sqrt(np.nansum(X ** 2, 0) / X.shape[0])

        # first get the winners
        if hasattr(self.winner_model, "support_"): # if it has support_ then it's already been done 
            self.winner_model.n_featrues_to_select = self.n_features_to_select # update the number of features to be selected
            self.feature_mask = self.winner_model.set_support_(X, Y, self.winner_model.support_)
        else: # then it hasn't been done, so do it
            self.feature_mask = self.winner_model.set_support_(X, Y)
            
        # loop over voxels and fit ridge
        wnta_coef = np.zeros((Y.shape[1], X.shape[1]))
        for vox in range(Y.shape[1]):

            if np.any(Y[:, vox]):
                # get the selected features for the current voxel
                selected_vox = list(np.where(self.feature_mask[vox, :] == 1)[0])

                ## use the selected featuers to do a ridge regression 
                X_selected = X[:, selected_vox]

                ### scale X_selected
                scale_ = np.sqrt(np.nansum(X_selected ** 2, 0) / X_selected.shape[0])
                Xs = X_selected / scale_
                Xs = np.nan_to_num(Xs) # there are 0 values after scaling

                # print(f"doing ridge regression")
                super(Ridge, self).fit(Xs, Y[:, vox])

                # fill in the elements of the coef
                wnta_coef[vox, selected_vox] = self.coef_

        # set the coef_ attribute
        self.coef_ = wnta_coef

    def predict(self, X):
        Xs = X / self.scale_
        Xs = np.nan_to_num(Xs) # there are 0 values after scaling
        return Xs @ self.coef_.T  # weights need to be transposed (throws error otherwise)
