import os 
import pandas as pd
import numpy as np
import re
import glob

import seaborn as sns
# import matplotlib.pyplot as plt
from collections import MutableMapping
from collections import defaultdict

from connectivity.constants import Dirs
from connectivity import io

"""
Created on Wed 26 13:31:34 2020
Visualization routine for connectivity models

@author: Maedbh King and Ladan Shahshahani
"""

class DataManager:

    def __init__(self):
        pass

    def _load_json_file(self, json_fname):
        # load json file (contains eval params)
        return io.read_json(json_fname)

    def _load_hdf5_file(self, hdf5_fname):
        # load eval file (contains eval outputs)
        return io.read_hdf5(hdf5_fname)
        
    def _convert_flatten(self, data_dict, parent_key = '', sep ='_'): 
        """ conversion of nested dictionary into flattened dictionary
        """
        items = [] 
        for k, v in data_dict.items(): 
            new_key = parent_key + sep + k if parent_key else k 
    
            if isinstance(v, MutableMapping): 
                items.extend(self._convert_flatten(v, new_key, sep = sep).items()) 
            else: 
                items.append((new_key, v)) 
        return dict(items)

    def _convert_to_dataframe(self, data_dict):
        cols_to_explode =  ['eval_splits', 'lambdas', 'eval_subjects'] 

        dataframe = pd.DataFrame.from_dict(data_dict)                                                                                                                                       
        for col in cols_to_explode: 
            dataframe = dataframe.explode(col)

        return dataframe

    def _flatten_nested_dict(self, data_dict):
        default_dict = defaultdict(list)

        for k,v in data_dict.items():
            if type(data_dict[k]) == dict: 
                v = self._convert_flatten(data_dict=data_dict[k])
                for kk,vv in v.items():
                    default_dict[kk].append(vv)
            else:
                default_dict[k].append(v)

        return default_dict
    
    def get_filenames(self):
        all_files = {}
        for file in self.eval_files:
            all_files[file] = glob.glob(os.path.join(self.dirs.CONN_EVAL_DIR, f'*{file}*.json'))

        return all_files
    
    def read_files_to_dataframe(self, files):

        for model in files.keys():
            
            repeat_files = files[model]

            for file in repeat_files:
                
                # load json and hdf5 files
                json_dict = self._load_json_file(json_fname=file)
                hdf5_dict = self._load_hdf5_file(hdf5_fname=file.replace('json', 'h5'))

                # flatten nested json dict
                json_dict = self._flatten_nested_dict(data_dict = json_dict)

                # convert json and hdf5 to dataframes
                df_json = self._convert_to_dataframe(data_dict = json_dict)
                df_hdf5 = pd.DataFrame.from_dict(hdf5_dict)

                keyboard

                # merge dataframes
                df_concat = df_json.merge(df_hdf5)

        return df_json, df_hdf5, df_concat

class Graphs(DataManager):

    def __init__(self, eval_files = ['tesselsWB162_grey_nan_l2_regress'], eval_on = 'sc2', glm = 7):
        self.eval_files = eval_files
        self.eval_on = eval_on
        self.glm = glm

        self.dirs = Dirs(study_name = self.eval_on, glm = self.glm)

    def load_dataframe(self):
        # get json files corresponding to`eval_files`
        fnames = self.get_filenames()

        df_json, df_hdf5, df_concat = self.read_files_to_dataframe(files = fnames)

        return df_json, df_hdf5, df_concat

    def plot_prediction_group(self, eval_params, eval_outputs):
        pass

    def plot_prediction_indiv(self, eval_params, eval_outputs):
        pass
        
class Maps(DataManager):
    
    def __init__(self):
        pass