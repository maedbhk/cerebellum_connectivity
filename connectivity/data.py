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
import SUITPy as suit
import nibabel as nib

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

    def __init__(self, experiment="sc1", glm="glm7", roi="cerebellum_suit", subj_id="s03"):
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
        self.CN = cio._convertobj(file, "CN")
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
        d = {
            "TN": self.TN,
            "CN": self.CN,
            "sess": self.sess,
            "run": self.run,
            "inst": self.inst,
            "task": self.task,
            "cond": self.cond,
        }
        return pd.DataFrame(d)


    def get_info_run(self):
        """Returns info for a typical run only."""
        info = self.get_info()
        return info[info.run == 1]


    def get_data(self, averaging="sess", weighting=True, subset=None):
        """Get the data using a specific aggregation.

        Args:
            averaging (str): sess (within each session); None (no averaging); exp (across whole experiment)
            weighting (bool): Should the betas be weighted by X.T * X?
            subset (index-like): Indicate the subset of regressors that should be considered
        Returns:
            data (np.array): aggregated data
            data_info (pandas dataframe): dataframe for the aggregated data
        """
        # check that mat is loaded
        try:
            hasattr(self, "data")
        except ValueError:
            print("Please run load_mat before returning data")

        num_runs = max(self.run)
        num_reg = sum(self.run == 1)
        # N = self.sess.shape[0]
        info = self.get_info()

        # Create unique ID for each regressor for averaging and subsetting it
        info["id"] = np.kron(np.ones((num_runs,)), np.arange(num_reg)).astype(int)
        if subset is None:
            subset = np.arange(num_reg)
        elif subset.dtype == "bool":
            subset = subset.nonzero()[0]

        # Different ways of averaging
        if averaging == "sess":
            X = matrix.indicator(info.id + (self.sess - 1) * num_reg)
            data = np.linalg.solve(X.T @ X, X.T @ self.data)
            data_info = info[np.logical_or(info.run == 1, info.run == 9)]
        elif averaging == "exp":
            X = matrix.indicator(info.id)
            data = np.linalg.solve(X.T @ X, X.T @ self.data)
            data_info = info[info.run == 1]
        elif averaging == "none":
            data_info = info
            data = self.data
        else:
            raise (NameError("averaging needs to be sess, exp, or none"))

        # data_infoubset the data
        indx = np.in1d(data_info.id, subset)
        data = data[indx, :]
        data_info = data_info[indx]

        # Now weight the different betas by the variance that they predict for the time series.
        # This also removes the mean of the time series implictly.
        # Note that weighting is done always on the average regressor structure, so that regressors still remain exchangeable across sessions
        if weighting:
            XXm = np.mean(self.XX, 0)
            XXm = XXm[subset, :][:, subset]  # Get the desired subset only
            XXs = scipy.linalg.sqrtm(XXm)  # Note that XXm = XXs @ XXs.T
            for r in np.unique(data_info["run"]):  # WEight each run/session seperately
                idx = data_info.run == r
                data[idx, :] = XXs @ data[idx, :]

        # There are NaN values in cerebellum_suit, which causes problems later on in fitting model.
        if self.roi == "cerebellum_suit":
            data = np.nan_to_num(data)

        return data, data_info


def convert_to_vol(data, xyz, voldef):
    """
    This function converts 1D numpy array data to 3D vol space, and returns nib obj
    that can then be saved out as a nifti file
    Args:
        data (list or 1d numpy array)
            voxel data, shape (num_vox, )
        xyz (nd-array)
            3 x P array world coordinates of voxels
        voldef (nib obj)
            nib obj with affine
    Returns:
        list of Nib Obj

    """
    # get dat, mat, and dim from the mask
    dim = voldef.shape
    mat = voldef.affine

    # xyz to ijk
    ijk = suit.flatmap.coords_to_voxelidxs(xyz, voldef)
    ijk = ijk.astype(int)

    vol_data = np.zeros(dim)
    vol_data[ijk[0],ijk[1],ijk[2]] = data

    # convert to nifti
    nib_obj = nib.Nifti1Image(vol_data, mat)
    return nib_obj


def convert_cerebellum_to_nifti(data):
    """
    Args:
        data (np-arrray): N x 6937 length data array
        or 1-d (6937,) array
    Returns:
        nifti (List of nifti1image): N output images
    """
    # Load the region file
    dirs = const.Dirs(exp_name="sc1")
    group_dir = os.path.join(dirs.reg_dir, 'data','group')
    reg_file = os.path.join(group_dir,'regions_cerebellum_suit.mat')
    region = cio.read_mat_as_hdf5(fpath=reg_file)["R"]

    # NII File for volume definition
    suit_file = os.path.join(group_dir,'cerebellarGreySUIT3mm.nii')
    nii_suit = nib.load(suit_file)

    # Map the data
    nii_mapped = []
    if data.ndim == 2:
        for i in range(data.shape[0]):
            nii_mapped.append(convert_to_vol(data[i],region.data.T,nii_suit))
    elif data.ndim ==1:
        nii_mapped.append(convert_to_vol(data,region.data.T,nii_suit))
    else:
        raise(NameError('data needs to be 1 or 2-dimensional'))
    return nii_mapped


def convert_cortex_to_gifti(data, atlas, hemisphere):
    """
    Args:
        data (np-arrray): 1 x 39876 length data array
        atlas (str): cortical atlas name (e.g. tessels0162)
        hemisphere (str): 'R' or 'L'
    Returns:
        gifti img
    """
    dirs = const.Dirs()

    # get texture
    gii_path = os.path.join(dirs.reg_dir, 'data', 'group', f'{atlas}.{hemisphere}.label.gii')
    gii_data = nib.load(gii_path)
    texture = gii_data.darrays[0].data[:]

    # get start and end labels
    start_label = texture[texture!=0].min()
    end_label = texture[texture!=0].max()
    labels_all = np.arange(1, len(data)+1)

    # get relevant data
    data_hemi = data[start_label-1:end_label]
    labels = labels_all[start_label-1:end_label]

    # get texture
    texture = texture.astype(float)
    for vert in np.arange(len(texture)):
        label = texture[vert]
        if label != 0:
            texture[vert] = data_hemi[labels==label]
    
    # get anatomical structure
    if hemisphere=="R":
        anatomical_struct = 'CortexRight'
    elif hemisphere=="L":
        anatomical_struct = 'CortexLeft'

    # return  gifti img
    return nio.make_func_gifti(data=texture.reshape(len(texture),1), anatomical_struct=anatomical_struct)

def get_distance_matrix(roi):
    """
    Args:
        roi (string)
            Region of interest ('cerebellum_suit','tessels0042','yeo7')
    Returns
        distance (numpy.ndarray)
            PxP array of distance between different ROIs / voxels
    """
    dirs = const.Dirs(exp_name="sc1")
    group_dir = os.path.join(dirs.reg_dir, 'data','group')
    if (roi=='cerebellum_suit'):
        reg_file = os.path.join(group_dir,'regions_cerebellum_suit.mat')
        region = cio.read_mat_as_hdf5(fpath=reg_file)["R"]
        D = eucl_distance(region.data)
    else:
        Dist = []
        parcels = [] 
        for h,hem in enumerate(['L','R']):
            # Load the corresponding label file 
            label_file = os.path.join(group_dir,roi + '.' + hem + '.label.gii')
            labels = nib.load(label_file)
            roi_label = labels.darrays[0].data

            # Load the spherical gifti 
            sphere_file = os.path.join(group_dir,'fs_LR.32k.' + hem + '.sphere.surf.gii')
            sphere = nib.load(sphere_file)
            vertex = sphere.darrays[0].data
        
            # Loop over the regions > 0 and find the average coordinate 
            parcels.append(np.unique(roi_label[roi_label>0]))
            num_parcels = parcels[h].shape[0]
            coord = np.zeros((num_parcels,3))
            for i,par in enumerate(parcels[h]):
                if par>0:
                    coord[i,:] = vertex[roi_label==par,:].mean(axis=0)
            
            # Calculate the distances over these 
            Dist.append(eucl_distance(coord))
        # Concatinate these.  
        num_regions = Dist[0].shape[0]+Dist[1].shape[0]
        D = np.zeros((num_regions,num_regions))
        # Still needs fixing: 
        # D[parcels[0],parcels[0]]=Dist[h]
        # D[parcels[1],parcels[1]]=Dist[h]
    return D

def eucl_distance(coord):
    """
    Calculates euclediand distances over some cooordinates 
    Args:
        coord (ndarray)
            Nx3 array of x,y,z coordinates 
    Returns: 
        dist (ndarray) 
            NxN array pf distances 
    """
    num_points = coord.shape[0]
    D = np.zeros((num_points,num_points))
    for i in range(2):
        D = D + (coord[:,i].reshape(-1,1)-coord[:,i])**2
    return np.sqrt(D)