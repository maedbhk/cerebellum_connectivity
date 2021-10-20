# import libraries and packages
import os
import pandas as pd
import numpy as np
import deepdish as dd
import scipy
import h5py
from SUITPy import flatmap
import nibabel as nib

# Import module as such - no need to make them a class
import connectivity.constants as const
import connectivity.io as cio
import connectivity.matrix as matrix
import connectivity.nib_utils as nio

"""Main module for getting data to be used for running connectivity models.

   @authors: Maedbh King, Ladan Shahshahani, Jörn Diedrichsen

  Typical usage example:
  data = Dataset('sc1','glm7','cerebellum_suit','s02')
  data.load_mat() # Load from Matlab 
  X, INFO = data.get_data(averaging="sess") # Get numpy 

  Group averaging: 
  data = Dataset(subj_id = const.return_subjs) # Any list of subjects will do 
  data.load_mat()                             # Load from Matlab
  data.average_subj()                         # Average 

  Saving and loading as h5: 
  data.save(dataname="group")     # Save under new data name (default = subj_id)
  data = Dataset('sc1','glm7','cerebellum_suit','group')
  data.load()

"""

class Dataset:
    """Dataset class, holds betas for one region, one experiment, one subject for connectivity modelling.

    Attributes:
        exp: A string indicating experiment.
        glm: A string indicating glm.
        roi: A string indicating region-of-interest.
        subj_id: A string for subject id - if the subj_id is a list of strings, the data will be averaged across these subjects. Thus, to get group-averaged data, set subj_id = const.return_subj
        data: None
    """

    def __init__(self, experiment="sc1", glm="glm7", roi="cerebellum_suit", subj_id="s02"):
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

        # For a single subject - make it a list 
        if type(self.subj_id) is not list:
            subj_id = [self.subj_id]
        else: 
            subj_id = self.subj_id
        num_subj = len(subj_id)
        # Iterate over all subjects
        for i,s in enumerate(subj_id): 
            fdir = dirs.beta_reg_dir / s
            file = h5py.File(fdir / fname, "r")

            # Store the data in betas x voxel/rois format
            d = np.array(file["data"]).T
            if (self.data is None):
                self.data = np.zeros((num_subj,d.shape[0],d.shape[1]))
            self.data[i,:,:] = d
            # this is the row info
            self.XX = np.array(file["XX"])
            self.TN = cio._convertobj(file, "TN")
            self.CN = cio._convertobj(file, "CN")
            self.cond = np.array(file["cond"]).reshape(-1).astype(int)
            self.inst = np.array(file["inst"]).reshape(-1).astype(int)
            self.task = np.array(file["task"]).reshape(-1).astype(int)
            self.sess = np.array(file["sess"]).reshape(-1).astype(int)
            self.run = np.array(file["run"]).reshape(-1).astype(int)
        
        # Remove third dimension if single subject
        if num_subj==1: 
            self.data = self.data.reshape(d.shape)
        return self

    def load_h5(self):
        """
            Load the content of a data set object from a hpf5 file.
            Returns:
                Data set object
        """
        dirs = const.Dirs(exp_name=self.exp, glm=self.glm)
        fname = "Y_" + self.glm + "_" + self.roi + ".h5"
        fdir = dirs.beta_reg_dir / self.subj_id

        a_dict = dd.io.load(fdir / fname)
        for key, value in a_dict.items():
            setattr(self,key,value)
        return self

    def load(self):
        """
            Utility function to first try to load subjects as h5 file
            and then as a mat file
        """
        try: 
            self.load_h5()
        except:
            self.load_mat()
        return self

    def save(self, dataname = None, filename=None):
        """Save the content of the data set in a dict as a hpf5 file.

        Args:
            dataname (str): default is subj_id - but can be set for group data
            filename (str): by default will be set to something automatic 
        Returns:
            saves dict to disk
        """
        if filename is None:
            if dataname is None: 
                if type(self.subj_id) is list: 
                    raise(NameError('For group data need to set data name'))
                else: 
                    dataname = self.subj_id
            dirs = const.Dirs(exp_name=self.exp, glm=self.glm)
            fname = "Y_" + self.glm + "_" + self.roi + ".h5"
            fdir = dirs.beta_reg_dir / dataname
        dd.io.save(fdir / fname, vars(self), compression=None)

    def average_subj(self): 
        """
            Averages data across subjects if data is 3-dimensional
        """
        if self.data.ndim == 2: 
            raise NameError('data is already 2-dimensional')
        self.data = np.nanmean(self.data, axis = 0)
    
    def get_info(self):
        """Return info for data set in a dataframe."""
        d = {
            "TN": self.TN,
            # "CN": self.CN,
            "sess": self.sess,
            "run": self.run,
            "inst": self.inst,
            "task": self.task,
            "cond": self.cond,
        }
        df = pd.DataFrame(d)

        # get common tasks
        df['TN'] = df['TN'].str.replace('2', '')
        common_tasks = ['verbGeneration', 'spatialNavigation', 'motorSequence', 'nBackPic', 'visualSearch', 'ToM', 'actionObservation', 'rest']

        # split tasks into 'common' and 'unique'
        df.loc[df['TN'].isin(common_tasks), 'split'] = 'common'
        df.loc[~df['TN'].isin(common_tasks), 'split'] = 'unique'

        return df

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

        # Data should be imputed if there are nan values
        data = np.nan_to_num(data)

        return data, data_info

def convert_to_vol(
    data, 
    xyz, 
    voldef
    ):
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
    ijk = flatmap.coords_to_voxelidxs(xyz, voldef)
    ijk = ijk.astype(int)

    vol_data = np.zeros(dim)
    vol_data[ijk[0],ijk[1],ijk[2]] = data

    # convert to nifti
    nib_obj = nib.Nifti1Image(vol_data, mat)
    return nib_obj

def convert_cerebellum_to_nifti(
    data
    ):
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

def convert_cortex_to_gifti(
    data, 
    atlas,
    data_type='func',
    column_names=None,
    label_names=None,
    label_RGBA=None,
    hem_names=['L', 'R']
    ):
    """
    Args:
        data (np-array): 1d- (cortical regions,). must correspond to `hem_names`
        atlas (str): cortical atlas name (e.g. tessels0162)
        data_type (str): 'func' or 'label'. default is 'func'
        column_names (list or None): default is None
        label_names (list or None): default is None
        label_RGBA (list or None): default is None
        hem_names (list of str): default is ['L', 'R']
    Returns:
        List of gifti-img (left + right hemisphere)
        anatomical_structure (list of hemisphere names)
    """
    dirs = const.Dirs()
    anatomical_struct = ['CortexLeft','CortexRight']
    # get texture
    gifti_img = []
    for h,(hem,struct) in enumerate(zip(hem_names, anatomical_struct)):
        # Load the labels (roi-numbers) from the label.gii files
        gii_path = os.path.join(dirs.reg_dir, 'data', 'group', f'{atlas}.{hem}.label.gii')
        gii_data = nib.load(gii_path)
        labels = gii_data.darrays[0].data[:]

        # ensure that data is float
        data = data.astype(float)

        n_row = data.shape
        c_data = np.insert(data, 0, np.nan)
        # Fastest way: prepend a NaN for ROI 0 (medial wall)
        try:
            mapped_data = c_data[labels, None]
        except:
            idx = labels-n_row
            np.put_along_axis(idx, np.where(idx<0)[0], 0, axis=0)
            mapped_data = c_data[idx, None]

        if data_type=='func':
            gii = nio.make_func_gifti_cortex(
                data=mapped_data,
                anatomical_struct=struct,
                column_names=column_names)
        elif data_type=='label':
            gii = nio.make_label_gifti_cortex(
                data=mapped_data,
                anatomical_struct=struct,
                label_names=label_names,
                label_RGBA=label_RGBA)
        gifti_img.append(gii)
        
    return gifti_img, hem_names

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
        coord = region.data
    else:
        coordHem = []
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

            # To achieve a large seperation between the hemispheres, just move the hemispheres apart 50 cm in the x-coordinate
            vertex[:,0] = vertex[:,0]+(h*2-1)*500

            # Loop over the regions > 0 and find the average coordinate
            parcels.append(np.unique(roi_label[roi_label>0]))
            num_parcels = parcels[h].shape[0]
            coordHem.append(np.zeros((num_parcels,3)))
            for i,par in enumerate(parcels[h]):
                coordHem[h][i,:] = vertex[roi_label==par,:].mean(axis=0)

        # Concatinate these to a full matrix
        num_regions = max(map(np.max,parcels))
        coord = np.zeros((num_regions,3))
        # Assign the coordinates - note that the
        # Indices in the label files are 1-based [Matlab-style]
        # 0-label is the medial wall and ignored!
        coord[parcels[0]-1,:]=coordHem[0]
        coord[parcels[1]-1,:]=coordHem[1]

    # Now get the distances from the coordinates and return
    Dist = eucl_distance(coord)
    return Dist, coord

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
    for i in range(3):
        D = D + (coord[:,i].reshape(-1,1)-coord[:,i])**2
    return np.sqrt(D)

def read_suit_nii(nii_file):
    """
    takes in a atlas file name in suit space
    Args:
        altas_file  - nifti filename for the atlas
    Returns:
        region_number_suit - values from parcellation file in suit space
    """

    # Load the region file for cerebellum in suit space
    dirs = const.Dirs(exp_name="sc1")
    group_dir = os.path.join(dirs.reg_dir, 'data','group')
    reg_file = os.path.join(group_dir,'regions_cerebellum_suit.mat')
    region = cio.read_mat_as_hdf5(fpath=reg_file)["R"]

    # get the coordinates of the cerebellum suit
    coords = region.data.T

    # load in the vol for the atlas file
    vol_def = nib.load(nii_file)

    # convert to voxel space
    ijk = flatmap.coords_to_voxelidxs(coords,vol_def).astype(int)

    indices = ijk.T

    # get the volume data
    vol_data = vol_def.get_fdata()

    # use indices to sample from vol_data
    data = vol_data[indices[:, 0], indices[:, 1], indices[:, 2]]
    return data

def average_by_roi(data, region_number_suit):
    """
    Takes in a matrix containing voxels in suit space and the value of the parcel (output from read_suit_nii)
    and calculate the average for each roi
    Args:
        data                - data in suit space (NxP)
        region_number_suit  - parcel vector in suit space (np.ndarray)
    Returns:
        data_mean_roi       - numpy array with mean within each roi (to be used as input to convert_cerebellum_to_nifti)
    """

    # reshape data into NxP dims
    try: 
        num_cols, num_vox = data.shape
    except: 
        data = np.reshape(data, (1, len(data)))

    # find region numbers
    region_number_suit = region_number_suit.astype("int")
    region_numbers = np.unique(region_number_suit)
    num_reg = len(region_numbers)
    # loop over regions and calculate mean for each
    # initialize the data array

    data_mean_roi = np.zeros((data.shape[0],num_reg))
    for r in range(num_reg):
        # get the indices of voxels in suit space
        reg_index = region_number_suit == region_numbers[r]

        # get data for the region
        reg_data = np.nanmean(data[:,reg_index], axis=1) # was np.mean

        # fill in data_roi
        data_mean_roi[:, r] = reg_data

    return data_mean_roi, region_numbers
