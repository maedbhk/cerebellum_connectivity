# import packages
from pathlib import Path
import scipy.io as sio 
import pandas as pd
import h5py
import deepdish as dd 
import shutil

"""
General purpose utils for importing mat files and 
outputting as HDF5 file objects
@ author: Maedbh King
"""

def read_mat_as_hdf5(fpath):
    """ imports mat files and returns HDF5 file object
        Args: 
            fpath (str): full path to mat file
        Returns: 
            mat file as HDF5 object
    """
    try: 
        # try loading with h5py (mat files saved as -v7.3)
        f = h5py.File(fpath, 'r')
        hf5_file = fpath.replace('.mat', '.h5')
        shutil.copyfile(fpath, hf5_file)
        return read_hdf5(hf5_file)

    except OSError: 
        # load mat struct with scipy
        data_dict = sio.loadmat(fpath, struct_as_record = False, squeeze_me = True)

        # save dict to hdf5
        hf5_file = fpath.replace('.mat', '.h5')
        save_dict_as_hdf5(fpath = hf5_file, data_dict = data_dict)
        return dd.io.load(hf5_file)

def read_hdf5(fpath):
    """ reads in HDF5 file
        Args: 
            fpath (str): full path to .h5 file
        Returns
            HDF5 object
    """
    return h5py.File(fpath, 'r')

def save_dict_as_hdf5(fpath, data_dict):
    """ saves dict as HDF5
        Args:
            fpath (str): save path for HDF5 file (.h5)
            dict (dict): python dict
        Returns: 
            saves dict to disk as HDF5 file obj
    """
    dd.io.save(fpath, data_dict, compression = None) 

def convert_to_dataframe(file_obj, cols):
    """ reads in datasets from HDF5 and saves out pandas dataframe
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
            col_values = file_obj[col].value.flatten().astype(int)
        except: 
            col_values = _convertobj(file_obj = file_obj, key = col)
        dict_all[col] = col_values

    dataframe = pd.DataFrame.from_records(dict_all)

    return dataframe

def _convertobj(file_obj, key):
    dataset = file_obj[key]
    tostring = lambda obj: ''.join(chr(i) for i in obj[:])
    return [tostring(file_obj[val]) for val in dataset.value.flatten()]
