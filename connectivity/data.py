# import libraries and packages
import os
import pandas as pd
import numpy as np
import re
import deepdish as dd
import copy
import scipy
import scipy.io as sio
from collections import defaultdict
import h5py

# Import module as such - no need to make them a class
import connectivity.constants as const
import connectivity.io as cio
from connectivity.helper_functions import AutoVivification
import connectivity.matrix as matrix
from numpy.linalg import solve

"""Main module for getting data to be used for running connectivity models.

   @authors: Maedbh King, Ladan Shahshahani, Jörn Diedrichsen  

  Typical usage example:

  cereb_data = Dataset(roi='cerebellum_grey')
  data, info = cereb_data.get_data()
"""


class Dataset:
    """Dataset class, holds betas for one region, one experiment, one subject for connectivity modelling.

    Attributes:
        exp: A string indicating experiment.
        glm: A string indicating glm.
        roi: A string indicating region-of-interest.
        subj_id: A string for subject id.
        data: None
    """

    def __init__(self, experiment="sc1", glm="glm7", roi="cerebellum_grey", subj_id="s03"):
        """Inits Dataset."""
        self.exp = experiment
        self.glm = glm
        self.roi = roi
        self.subj_id = subj_id
        self.data = None

    def load_mat(self):
        """Reads a data set from the Y_info file and corresponding GLM file from matlab."""
        dirs = const.Dirs(exp_name=self.exp, glm=self.glm)
        fname = "Y_" + self.glm + "_" + self.roi + ".mat"
        fdir = dirs.beta_reg_dir / self.subj_id
        file = h5py.File(fdir / fname, "r")
        # Store the data in betas x voxel/rois format
        self.data = np.array(file["data"]).T
        # this is the row info
        self.XX = np.array(file["XX"])
        self.TN = cio._convertobj(file, "TN")
        self.cond = np.array(file["cond"]).reshape(-1).astype(int)
        self.inst = np.array(file["inst"]).reshape(-1).astype(int)
        self.task = np.array(file["task"]).reshape(-1).astype(int)
        self.sess = np.array(file["sess"]).reshape(-1).astype(int)
        self.run = np.array(file["run"]).reshape(-1).astype(int)
        return self

    def save(self, filename=None):
        """Save the content of the data set in a dict as a hpf5 file.

        Args:
            filename (str): default is None.
        Returns:
            saves dict to disk
        """
        if filename is None:
            dirs = const.Dirs(study_name=self.exp, glm=self.glm)
            fname = "Y_" + self.glm + "_" + self.roi + ".h5"
            fdir = dirs.beta_reg_dir / self.subj_id

        dd.io.save(fdir / fname, vars(self), compression=None)

    def load(self, filename=None):
        """Load the content of a data set object from a hpf5 file.
        Args:
            filename (str): default is None.
        Returns:
            returns dict from hpf5.
        """
        if filename is None:
            dirs = Dirs(study_name=self.exp, glm=self.glm)
            fname = "Y_info_" + self.glm + "_" + self.roi + ".h5"
            fdir = dirs.BETA_REG_DIR / dirs.BETA_REG_DIR / self.subj_id

        return dd.io.load(fdir / fname, self, compression=None)

    def get_info(self):
        """Return info for data set in a dataframe."""
        self.info = {
            "TN": self.TN,
            "sess": self.sess,
            "run": self.run,
            "inst": self.inst,
            "task": self.task,
            "cond": self.cond,
        }
        return self.info

    def get_info_run(self):
        """Returns info for a typical run only."""
        info = self.get_info()
        return info[info.run == 1]

    def get_subset(self, subset):
        """returns regressor indices for subsetting the data.

        Args:
            subset():
        Returns:
            np array of subset indices
        """

        num_reg = sum(self.run == 1)

        # Create unique ID for each regressor for averaging and subsetting it
        self.info["id"] = np.kron(np.ones((max(self.run),)), np.arange(num_reg)).astype(int)
        if subset is None:
            self.subset = np.arange(num_reg)
        elif subset.dtype == "bool":
            self.subset = subset.nonzero()[0]

        return self.subset

    def average(self, averaging):
        """Average the data by session, experiment, or no averaging.

        Args:
            averaging (str): options are 'sess', 'exp', 'none'
        Returns:
            Y_data (nd numpy array), df (pandas dataframe)
        """
        if averaging == "sess":
            X = matrix.indicator(self.info.id + (self.sess - 1) * sum(self.run == 1))
            Y_data = np.linalg.solve(X.T @ X, X.T @ self.data)
            df = self.info[np.logical_or(self.info.run == 1, self.info.run == 9)]
        elif averaging == "exp":
            X = matrix.indicator(self.info.id)
            Y_data = np.linalg.solve(X.T @ X, X.T @ self.data)
            df = self.info[self.info.run == 1]
        elif averaging == "none":
            df = self.info
            Y_data = self.data
        else:
            raise (NameError("averaging needs to be sess, exp, or none"))

        return Y_data, df

    def weight(self, data, runs):
        """Weight the betas by the variance they predict for the timeseries.

        Weighting is always done on the average regressor structure, so that
        regressors remain exchangeable. The mean is implicitly removed from the timeseries.

        Args:
            data (np.array):
            runs (np.array): array of run numbers
        Returns:

        """
        XXm = np.mean(self.XX, 0)
        XXm = XXm[self.subset, :][:, self.subset]  # Get the desired subset only
        XXs = scipy.linalg.sqrtm(XXm)  # Note that XXm = XXs @ XXs.T
        for run in runs:  # Weight each run/session seperately
            idx = df.run == run
            data[idx, :] = XXs @ data[idx, :]

        return data

    def get_data(self, averaging="sess", weighting=True, subset=None):
        """Get the data using a specific aggregation.

        Args:
            averaging (str): sess (within each session); None (no averaging); exp (across whole experiment)
            weighting (bool): Should the betas be weighted by X.T * X?
            subset (index-like): Indicate the subset of regressors that should be considered
        Returns:
            data (np.array): aggregated data
            info (pandas dataframe): dataframe for the aggregated data
        """
        # check that mat is loaded
        if not self.data:
            self.load_mat()

        # return info for dataset
        self.get_info()

        # return subset indices
        self.get_subset(subset=subset)

        # average data
        data, info = self.average(averaging=averaging)

        # Subset the data
        indx = np.in1d(info.id, self.subset)
        data = data[indx, :]
        info = info[indx]

        # weight data
        if weighting:
            data = self.weight(data=data, runs=np.unique(info.run))

        return data, info
