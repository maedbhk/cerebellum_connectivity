#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:41:37 2020
Data integration module for the connectivity project
@author: ladan
"""

# import packages
#import os                    # might be needed in handling file paths
import pandas as pd
import numpy as np
#import scipy as sp
import scipy.io as spio     # for .mat files saved with versions before 7
import mat73                 # for .mat files saved with version 7.3 and higher
import nibabel as nib        # to handle gifti files

# function definitions
def matImport(path2mfile, form = 'dict'):
    """
    matImport is used to import MATLAB's .mat file into python.
    What are the files we need to import?
    - a structure with the data used in modelling:
        * activity profiles
        * predicted time series
        * raw time series
        * residuals
    - a structure with the model's data: just in case the model was created in matlab
        * different models will have different fields?!
        * All models should have the W (regression parameter corresponding to connectivity weights)
        
    INPUTS: 
    - path2mfile: directory path where the mat file is saved.
    - form      : the output format. It can either be a dict or a pandas dataframe. The default is set to 'dict'
                  other option is 'dataframe'
    
    OUTPUTS:
    - mat    : variable containing the loaded mat file
    
    additional notes:
    some matfiles are nested structures. If they are saved with versions other than -7.3, scipy.io.loadmat
    should be used. However, this method cannot handle nested structures (and most of our mat files are 
    nested structures). To solve this problem, I am using the method proposed in:
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    Basically, this stackoverflow suggests using a modified version of the loadmat function.
    
    WARNING: This function cannot read SPM.mat files. SPM.mat fiels saved with 'v7.3' will raise an error!
    """
    
    # For mat files saved as version 7.3: use mat73 package
    # if you do not have mat73 package, pip install mat73
    # https://pypi.org/project/mat73/
    # For mat files saved as version 7 and before: use the modified version of scipy.io.loadmat
    
    try: # tries loading the mat files saved as versions before 7.3

        # These functions are copy-pasted from this thread in stackoverflow
        # https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print('loading mat file')
        def loadmat(path2mfile):
            '''
            this function should be called instead of direct spio.loadmat
            as it cures the problem of not properly recovering python dictionaries
            from mat files. It calls the function check keys to cure all entries
            which are still mat-objects
            '''
            data = spio.loadmat(path2mfile, struct_as_record=False, squeeze_me=True)
            return _check_keys(data)

        def _check_keys(dict):
            '''
            checks if entries in dictionary are mat-objects. If yes
            todict is called to change them to nested dictionaries
            '''
            for key in dict:
                if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                    dict[key] = _todict(dict[key])
            return dict        

        def _todict(matobj):
            '''
            A recursive function which constructs from matobjects nested dictionaries
            '''
            dict = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                    dict[strg] = _todict(elem)
                else:
                    dict[strg] = elem
            return dict
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        data_dict = loadmat(path2mfile)
        keys = list(data_dict.keys())
        for k in keys:
              if k.startswith('_'):
                data_dict.pop(k)

    except NotImplementedError: # tries importing mat fiels saved as -v7.3

        data_dict = mat73.loadmat(path2mfile) # a dictionary

    except:
        ValueError('could not read the mat file')
        
    
    if len(data_dict.keys()) == 1: # one-var structs
        key    = list(data_dict.keys()) # get the name of the variable
        D      = data_dict[key[0]]   # get the actual data: Y, B, ...
        D_dict = {}               # preallocating a new dictionary that will contain the data
        for k in D.keys():
            arr = np.array(D[k]) # making sure values are stored as numpy array
            # for cortical and cerebellar maps, with 2 dimensions, the numpy array will be 
            # converted to a nested list to be able to store it in the dataframe
            
            if form == 'dataframe':
                # if it is to be saved as a dataframe, maps (cortical/cerebellar) should be converted to 
                # nested lists to be able to store it in a column of the dataframe
                try: # if arr is 1-D, the next line will raise a ValueError.
                    [rc, cv] = arr.shape
                    bmap = []
                    for i in arr.reshape(-1,arr.shape[1],1):
                        bmap.append(i)
                    arrVal = bmap
                except ValueError: # to catch the value error
                    arrVal = arr
                
            elif form == 'dict':
                arrVal = arr
            
            D_dict[k] = arrVal
            
        if form == 'dataframe':
            mat = pd.DataFrame(D_dict)
        elif form == 'dict':
            mat = D_dict
            
    else: # multiple-var structs like SPM_info
        # for multiple-var structs it's pretty straight forward, you just convert it to a dataframe
        if form == 'dataframe':
            mat = pd.DataFrame(data_dict) 
        elif form == 'dict':
            mat = data_dict

    return mat

def giftiImport(path2gifti):
    """
    Is there already a function for that?????!!!!! if yes, use it!
    giftiImport is used to import GIFTI files into python. 
    These will be used for plotting maps on surfaces or reading in the info from gifti files if needed!
    
    INPUTS:
    - path2gifti: directory path where the gifti file is saved
    
    OUTPUTS:
    - G: variable containing info in the gifti file.
    
    Gifti documentation: https://nipy.org/nibabel/reference/nibabel.gifti.html#module-nibabel.gifti.giftiio
    """
    
    file_gifti = nib.load(path2gifti)
    
    # this is only returning a numpy array with all the data within the gifti image.
    # use dir() to get all the attributes of the data
    ## aggregate the data and converts it to numpy array
    G = np.array(file_gifti.agg_data())
    
    return G