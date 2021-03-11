# import packages
from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
import h5py
import deepdish as dd
import shutil
import json
import nibabel as nib
from nilearn.image import mean_img
import os
import SUITPy.flatmap as flatmap

"""General purpose module for loading and saving data.

@authors: Maedbh King 
"""


def read_mat_as_hdf5(fpath, save_to_h5=False):
    """imports mat files and returns HDF5 file object
    Args:
        fpath (str): full path to mat file
        save_to_h5 (bool): default is False
    Returns:
        mat file as HDF5 object
    """
    try:
        # try loading with h5py (mat files saved as -v7.3)
        f = h5py.File(fpath, "r")
        if save_to_h5:
            hf5_file = fpath.replace(".mat", ".h5")
            shutil.copyfile(fpath, hf5_file)
            f = h5py.File(hf5_file, "r")
        return f

    except OSError:
        # load mat struct with scipy
        f = sio.loadmat(fpath, struct_as_record=False, squeeze_me=True)

        # save dict to hdf5
        if save_to_h5:
            hf5_file = fpath.replace(".mat", ".h5")
            save_dict_as_hdf5(fpath=hf5_file, data_dict=f)
            f = dd.io.load(hf5_file)
        return f


def read_hdf5(fpath):
    """reads in HDF5 file
    Args:
        fpath (str): full path to .h5 file
    Returns
        HDF5 object
    """
    return dd.io.load(fpath)


def save_dict_as_JSON(fpath, data_dict):
    """saves dict as JSON
    Args:
        fpath (str): full path to .json file
        data_dict (dict): dict to save
    Returns
        saves out JSON file
    """
    with open(fpath, "w") as fp:
        json.dump(data_dict, fp, indent=4)


def read_json(fpath):
    """loads JSON file as dict
    Args:
        fpath (str): full path to .json file
    Returns
        loads JSON as dict
    """
    f = open(fpath)

    # returns JSON object as a dict
    return json.load(f)


def save_dict_as_hdf5(fpath, data_dict):
    """saves dict as HDF5
    Args:
        fpath (str): save path for HDF5 file (.h5)
        dict (dict): python dict
    Returns:
        saves dict to disk as HDF5 file obj
    """
    dd.io.save(fpath, data_dict, compression=None)


def convert_to_dataframe(file_obj, cols):
    """reads in datasets from HDF5 and saves out pandas dataframe
    assumes that there are no groups (i.e. no nested datasets)
    Args:
        fpath: full path to HDF5 file
        cols: list of cols to include in dataframe
    Returns:
        pandas dataframe
    """
    dict_all = {}
    for col in cols:
        try:
            col_values = file_obj[col][()].flatten().astype(int)
        except:
            col_values = _convertobj(file_obj=file_obj, key=col)
        dict_all[col] = col_values

    dataframe = pd.DataFrame.from_records(dict_all)

    return dataframe


def nib_load(fpath):
    return nib.load(fpath)


def nib_save(img, fpath):
    nib.save(img, fpath)


def convert_to_vol(data, xyz, mask):
    """
    This function converts 1D numpy array data to 3D vol space, and returns nib obj
    that can then be saved out as a nifti file
    Args:
        data (list of 1d numpy array): voxel data, shape (num_vox, )
        xyz (int): world coordinates corresponding to grey matter voxels for group
        mask (nib obj): nib obj with affine
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


def nib_mean(nib_objs):
    """get mean image of list of nib objs

    Args:
        nib_objs (list): list of nib objs
    Returns:
        mean nib obj
    """
    return mean_img(nib_objs)


def _convertobj(file_obj, key):
    """converts object reference for `key` in `file_obj`"""
    dataset = file_obj[key]
    tostring = lambda obj: "".join(chr(i) for i in obj[:])
    return [tostring(file_obj[val]) for val in dataset[()].flatten()]


def make_dirs(fpath):
    if not os.path.exists(fpath):
        print(f"creating {fpath}")
        os.makedirs(fpath)
