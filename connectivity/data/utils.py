# import packages
from pathlib import Path
import scipy.io as sio 
import pandas as pd
import h5py
import deepdish as dd 
import shutil

"""
Converts mat files into hdf5 objects
@ author: Maedbh King
"""

def read_mat_as_hdf5(fpath):
    """ imports mat files and returns HDF5 file object
        Args: 
            fpath: full path to mat file
        Returns: 
            mat file as HDF5 object
    """
    try: 
        # try loading with h5py (mat files saved as -v7.3)
        hf5_file = fpath.replace('.mat', '.h5')
        shutil.copyfile(fpath, hf5_file)
        return h5py.File(hf5_file, 'r')

    except OSError: 
        # load mat struct with scipy
        data_dict = sio.loadmat(fpath, struct_as_record=False, squeeze_me=True)

        # save dict to hdf5
        fpath = fpath.replace('.mat', '.h5')
        dd.io.save(fpath, data_dict, compression=None) 
        return dd.io.load(fpath)

def read_hdf5(fpath):
    """ reads in HDF5 file
        Args: 
            fpath: full path to .h5 file
        Returns
            HDF5 object
    """
    return h5py.File(fpath, 'r')

def convert_to_dataframe(file_obj):
    """ reads in datasets from HDF5 and saves out pandas dataframe
        assumes that there are no groups (i.e. no nested datasets)
        Args: 
            fpath: full path to HDF5 file
        Returns: 
            pandas dataframe
    """
    dict_all = {}
    for key in file_obj.keys():
        if key!='#refs#':
            try: 
                col_values = file_obj[key].value.flatten().astype(int)
            except: 
                col_values = _convertobj(file_obj = file_obj, key = key)
            dict_all[key] = col_values

    dataframe = pd.DataFrame.from_records(dict_all)

    return dataframe

def _convertobj(file_obj, key):
    dataset = file_obj[key]
    tostring = lambda obj: ''.join(chr(i) for i in obj[:])
    return [tostring(file_obj[val]) for val in dataset.value.flatten()]

