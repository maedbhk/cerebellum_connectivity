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
import flatmap
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
        d = {
            "TN": self.TN,
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


    def load_regions_cerebellum(self):
        """load cerebellum suit regions from mat file

        Returns:
            regions (dict)
        """
        dirs = const.Dirs(exp_name="sc1")
        os.chdir(os.path.join(dirs.reg_dir, "data/group"))
        regions = cio.read_mat_as_hdf5(fpath="regions_cerebellum_suit.mat")["R"]

        return regions


    def load_gifti_cerebellum(self, nib_obj, column_name):
        """maps nib obj to surface and returns gifti image

        Args:
            nib_obj (nib obj):
            column_names (str): column name
        Returns:
            gifti image
        """
        # map volume to surface
        surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT")

        # make gifti images
        gifti_img = flatmap.make_func_gifti(data=surf_data, column_names=column_name)

        return gifti_img


    def load_gifti_cortex(self, data_array, column_names = [], label_name = 'Icosahedron-162.32k', hemi = 'Left'):
        """
        maps an array containing data for the tessels and returns gifti image

        Args:
            map_array    -   array for which you want the gifti. (number_of_regions-by-number_of_variables)
                             A column in the gifti will be created for each column of the array
            column_names -   names of the columns in the gifti file. Default is empty
            roi_name     -   cortical roi name
            hemi         -   hemisphere: 'Left' or 'Right'
        Returns:
            gifti image
        """
        # get the shape of the data array
        ## r is the number of regions
        ## c is the number of maps (for pls for example, the map for each loading is in one column)
        [r, c] = data_array.shape

        # % Make column_names if empty
        if not column_names:
            for i in range(c):
                column_names.append('col_%d'% i)

        # preparing the gifti object
        # create the name of the structure
        anat_struct = f"Cortex{hemi}"

        # create the meta data for the gifti file
        img_meta = {}
        img_meta['AnatomicalStructurePrimary'] = anat_struct
        img_meta['encoding'] = 'XML_BASE64_GZIP'
        meta_ = nib.gifti.GiftiMetaData.from_dict(img_meta)

        # fix the label ???????????????????????
        img_label      = nib.gifti.gifti.GiftiLabel(key=0, red=1, green=1, blue=1, alpha=0)
        img_labelTable = nib.gifti.gifti.GiftiLabelTable()

        gifti_img = nib.GiftiImage(meta = meta_, labeltable = img_labelTable)

        # 1. load the label file. for tesselation it will be IcosahedranXXX.
        ## 1.1 the gifti file is in fs_LR directory
        dirs = const.Dirs(exp_name='sc1')
        fs_lr_dir = dirs.fs_lr_dir
        ## 1.2. load the gifti file
        label_gifti = nib.load(os.path.join(fs_lr_dir, f"{label_name}.{hemi}.label.gii"))
        ## 1.3. get the label assignments for each vertices
        vertex_label = label_gifti.agg_data()
        ## 1.4 delete the medial wall info
        label_names = label_gifti.labeltable.get_labels_as_dict()
        del label_names[0] # assuming that the medial wall is the first label
        ## 1.5 create label numbers
        ### labels for left hemi come first!
        if hemi == 'Left':
            label_nums = np.arange(len(label_names.keys()))
        elif hemi == 'Right':
            label_nums = np.arange(len(label_names.keys()), r)

        # 2. map the array to vertices
        data_vertex = np.nan * np.ones((vertex_label.shape[0], c)) # initializing output to be all nans
        for ic in range(c): # create a map corresponding to each label
            # get the map for the current component
            map_ic = data_array[:, ic]

            # preparing different fields of the gifti
            mmeta = {}
            mmeta['name']  = 'Name'
            mmeta['value'] = column_names[i]
            mmeta_ = nib.gifti.GiftiMetaData.from_dict(mmeta)
            coord_ = nib.gifti.gifti.GiftiCoordSystem(dataspace=0, xformspace=0, xform=None)

            # loop through regions
            ihemi = 0 # this index is always starting from zero and is used to read from the parcellation
            for i in label_nums:
                # get the value
                region_val = map_ic[i]

                # find the indices for the vertices of the parcel and
                # set their corresponding value equal to the value for that region
                data_vertex[vertex_label == ihemi+1, ic] = region_val

                gifti_img.add_gifti_data_array(nib.gifti.GiftiDataArray(data = data_vertex[:, ic], meta = mmeta_,
                                                intent = 'NIFTI_INTENT_NONE',
                                                datatype = 'NIFTI_TYPE_FLOAT32',
                                                coordsys = coord_))

                ihemi = ihemi +1

        return gifti_img


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

        # there are NaN values in cerebellum_suit, which causes problems later on in fitting model.
        if self.roi == "cerebellum_suit":
            data = np.nan_to_num(data)

        return data, data_info

def convert_to_vol(data, xyz, voldef):
    """
    This function converts 1D numpy array data to 3D vol space, and returns nib obj
    that can then be saved out as a nifti file
    Args:
        data (list of 1d numpy array): voxel data, shape (num_vox, )
        xyz (int): world coordinates corresponding to grey matter voxels for group
        voldef (nib obj): nib obj with affine
    Returns:
        list of Nib Obj

    """
    # get dat, mat, and dim from the mask
    dat = mask.get_fdata()
    dim = dat.shape
    mat = mask.affine

    # xyz to ijk
    ijk = flatmap.coords_to_voxelidxs(xyz, mask)
    ijk = ijk.astype(int)

    nib_objs = []
    for y in data:
        num_vox = len(y)
        # initialise xyz voxel data
        vol_data = np.zeros((dim[0], dim[1], dim[2]))
        for i in range(num_vox):
            vol_data[ijk[0][i], ijk[1][i], ijk[2][i]] = y[i]

        # convert to nifti
        nib_obj = nib.Nifti2Image(vol_data, mat)
        nib_objs.append(nib_obj)
    return nib_objs

def convert_cerebellum_to_nifti(data):
    """
    INPUT:
        data (np-arrray): 67xx length data array
    OUTPUT:
        nifti (nifti2image):

    """
def convert_cortex_to_gifti: