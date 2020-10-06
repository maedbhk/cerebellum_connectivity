import os 
import pandas as pd
import numpy as np
import re
import glob
import copy
from pathlib import Path
from dictdiffer import diff

import seaborn as sns
import matplotlib.pyplot as plt
from collections import MutableMapping
from collections import defaultdict
from functools import partial
import pprint

from nilearn import plotting
import nibabel as nib
from nilearn import surface

# import plotly.graph_objects as go

from connectivity.constants import Dirs, Defaults
from connectivity import io
from connectivity.data.prep_data import DataManager
from connectivity.visualization import image_utils

import warnings
warnings.filterwarnings('ignore')

"""
Created on Sep 05 07:03:32 2020
Visualization routine for connectivity models

@author: Maedbh King
"""

class Utils:
    def __init__(self):
        pass

    def _load_param_file(self, param_fname):
        # load json file (contains eval params)
        return io.read_json(param_fname)

    def _load_data_file(self, data_fname):
        # load eval file (contains eval outputs)
        return io.read_hdf5(data_fname)
        
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
    
    def _get_cerebellar_mask(self, mask, glm):
        """ converts cerebellar mask to nifti obj
            Args: 
                mask (str): make name
            Returns: 
                nifti obj of mask
        """
        dirs = Dirs(study_name='sc1', glm=glm)
        return nib.load(os.path.join(dirs.SUIT_ANAT_DIR, mask))

    def _make_dir(self, fpath):
        if not os.path.exists(fpath):
            os.makedirs(fpath)
    
    def get_all_files(self, fullpath, wildcard):
        return glob.glob(os.path.join(fullpath, wildcard))
    
    def read_to_dataframe(self, files):

        # get data for repeat models
        df_all = pd.DataFrame()
        df_merged = pd.DataFrame()
        for file in files:

            # load param file
            param_dict = self._load_param_file(param_fname=file)

            # only read summary data to dataframe (not voxel data)
            if not param_dict['eval_save_maps']:
            
                # load data file
                data_dict = self._load_data_file(data_fname=file.replace('json', 'h5'))

                # flatten nested json dict
                param_dict = self._flatten_nested_dict(data_dict=param_dict)

                try: 
                    df_param = self._convert_to_dataframe(data_dict=param_dict)
                    df_data = pd.DataFrame.from_dict(data_dict)
                    # merge param and data 
                    df_merged = df_param.merge(df_data)
                except: 
                    # add data dict to param_dict
                    param_dict.update(data_dict)
                    # convert json and hdf5 to dataframes
                    df_merged = self._convert_to_dataframe(data_dict=param_dict)

            # concat repated models
            df_all = pd.concat([df_all, df_merged], axis=0)
            df_all['eval_fname'] = Path(file).name

        # tidy up dataframe
        cols_to_stack = [col for col in df_all.columns if 'R_' in col]
        cols_to_stack.extend([col for col in df_all.columns if 'S_' in col])
        df1 = pd.concat([df_all]*len(cols_to_stack)).reset_index(drop=True)
        df2 = pd.melt(df_all[cols_to_stack]).rename({'variable': 'eval_type', 'value': 'eval'}, axis=1)
        df_all = pd.concat([df1, df2], axis=1)

        return df_all

class PlotPred(Utils):

    def __init__(self, eval_name=['tesselsWB162_grey_nan_l2_regress'], eval_on=['sc1', 'sc2'], glm=7, 
                eval_sparse_matrix=[False], train_X_file_dir='encoding', eval_X_file_dir='encoding', train_avg='run',
                eval_avg='run', train_incl_inst=True, eval_incl_inst=True, lambdas=[1,2,6,15,36,88,215,527,1292,3162],
                eval_good_vox=True, train_scale=True, eval_scale=True, train_stim='cond', train_mode=['crossed'],
                eval_stim='cond'):
        """ Initialises PlotPred class. Inherits functionality from Utils class
            Args: 
                eval_name (list of str): <roi1>_<roi2>_<model>. default is ['tesselsWB162_grey_nan_l2_regress']
                eval_on (list of str): study name(s). default is ['sc1', 'sc2']
                glm (int): glm name. default is 7
                eval_sparse_matrix (bool): default is false
                train_X_file_dir (str): default is 'encoding', another option is 'beta_roi'
                eval_X_file_dir (str): default is 'encoding', another option is 'beta_roi'
                train_avg (str): default is 'run', another option is 'sess'
                eval_avg (str): default is 'run', another option is 'sess'
                train_incl_inst (bool): default is True
                eval_incl_inst (bool): default is True
        """
        self.eval_name = eval_name
        self.eval_on = eval_on
        self.glm = glm
        self.eval_sparse_matrix = eval_sparse_matrix
        self.train_X_file_dir = train_X_file_dir
        self.eval_X_file_dir = eval_X_file_dir
        self.train_avg = train_avg
        self.eval_avg = eval_avg
        self.train_incl_inst = train_incl_inst
        self.eval_incl_inst = eval_incl_inst
        self.lambdas = lambdas
        self.train_scale = train_scale
        self.train_mode = train_mode
        self.eval_scale = eval_scale
        self.eval_good_vox = eval_good_vox
        self.train_stim = train_stim
        self.eval_stim = eval_stim
        self.config = {'train_incl_inst': train_incl_inst, 'glm': glm, 'eval_name': eval_name,
                       'train_X_file_dir': train_X_file_dir, 'train_avg': train_avg,
                       'train_scale': train_scale, 'train_stim': train_stim, 'train_mode': train_mode,
                       'eval_scale': eval_scale, 'eval_sparse_matrix': eval_sparse_matrix, 'eval_stim': eval_stim, 'eval_X_file_dir': eval_X_file_dir, 
                       'eval_avg': eval_avg, 'eval_on': eval_on, 
                       'eval_incl_inst': eval_incl_inst, 'eval_good_vox': eval_good_vox,
                       'lambdas': lambdas}

    def load_dataframe(self):
        """ loads dataframe containing data and model and eval params for `eval_name` and `eval_on`
            loads in data for all repeats of `eval_name`
        """
        # loop over exp
        df_all = pd.DataFrame()
        for exp in self.eval_on:

            for eval_name in self.eval_name:

                self.dirs = Dirs(study_name=exp, glm=self.glm)

                # get filenames for `eval_name` and for `exp`
                fnames = self.get_all_files(fullpath=self.dirs.CONN_EVAL_DIR, wildcard=f'*{eval_name}*.json')

                # read data to dataframe
                df_all = pd.concat([df_all, self.read_to_dataframe(files=fnames)])
                df_all['eval_name'] = eval_name

        # filter dataframe with based on args
        df_all = df_all.loc[(df_all['train_X_file_dir']==self.train_X_file_dir) & (df_all['train_avg']==self.train_avg) & 
                (df_all['eval_avg']==self.eval_avg) & (df_all['train_incl_inst']==self.train_incl_inst) & 
                (df_all['eval_incl_inst']==self.eval_incl_inst)]

        # filter lambdas
        df_all = df_all[df_all['lambdas'].isin(self.lambdas)]

        # filter sparse matrix
        df_all = df_all[df_all['eval_sparse_matrix'].isin(self.eval_sparse_matrix)]

        # filter eval_X_file_dir
        df_all = df_all[df_all['eval_name'].isin(self.eval_name)]

        # filter train_mode
        df_all = df_all[df_all['train_mode'].isin(self.train_mode)]

        return df_all

    def plot_lineplot(self, dataframe,
                        x='lambdas', 
                        y='eval', 
                        hue='eval_type', 
                        filter_eval=['R_y', 'R_pred_cv', 'R_pred_ncv'], 
                        forloop='eval_X_roi',
                        plot_params=True,
                        save_plot=True):
        """ plots predictions for `eval_name`
            Args: 
                dataframe (pandas dataframe): output from `load_dataframe`
                x (str): data to plot on x axis. default is 'lambdas'
                y (str): data to plot on y axis. default is 'eval'
                hue (str): option to split data. default is 'eval_type'
                forloop (str): produce multiple plots, default is 'eval_X_roi'
                filter_eval (list of str): filter evals. default is ['R_y', 'R_pred_crossed', 'R_pred_uncrossed']
                plot_params (bool): plot the model and eval params
        """
        if forloop:
            for loop_name in np.unique(dataframe[forloop]):

                # filter dataframe for eval
                dataframe_filter = dataframe.query(f'{forloop}=="{loop_name}" and eval_type=={filter_eval}')
                # train_on = np.unique(dataframe_filter['train_on'])[0]

                sns_plot = sns.factorplot(x=x, y=y, hue=hue, data=dataframe_filter, size=4, aspect=2)
                plt.xlabel(x, fontsize=20),
                plt.ylabel('R', fontsize=20)
                plt.title(f'{loop_name}', fontsize=20);
                plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
                # plt.ylim(bottom=.7, top=1.0)

                plt.show()
                # save plot
                sns_plot.savefig(os.path.join(self.dirs.FIGURES, f'line-{loop_name}.png'))
        else:
            # filter dataframe for eval
            dataframe_filter = dataframe.query(f'eval_type=={filter_eval}')

            sns_plot = sns.factorplot(x=x, y=y, hue=hue, data=dataframe_filter, size=4, aspect=2)
            plt.xlabel(x, fontsize=20),
            plt.ylabel('R', fontsize=20)
            plt.title(f'{self.eval_name[0]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
            # plt.ylim(bottom=.7, top=1.0)

            plt.show()

            eval_name = self.eval_name[0]
            sns_plot.savefig(os.path.join(self.dirs.FIGURES, f'line-{eval_name}-{x}-{filter_eval}-{hue}.png'))

        # optionally plot model and eval params
        if plot_params:
            # pprint.pprint(self._get_config_params(), compact=True)
            pprint.pprint(self.config, compact=True)

    def plot_barplot(self, dataframe,
                        x='lambdas', 
                        y='eval', 
                        hue='eval_type', 
                        filter_eval=['R_y', 'R_pred_cv', 'R_pred_ncv'], 
                        forloop='eval_X_roi',
                        plot_params=True,
                        save_plot=True):
        """ plots predictions for `eval_name`
            Args: 
                dataframe (pandas dataframe): output from `load_dataframe`
                x (str): data to plot on x axis. default is 'lambdas'
                y (str): data to plot on y axis. default is 'eval'
                hue (str): option to split data. default is 'eval_type'
                forloop (str): produce multiple plots, default is 'eval_X_roi'
                filter_eval (list of str): filter evals. default is ['R_y', 'R_pred_crossed', 'R_pred_uncrossed']
                plot_params (bool): plot the model and eval params
        """
        if forloop:
            for loop_name in np.unique(dataframe[forloop]):

                # filter dataframe for eval
                dataframe_filter = dataframe.query(f'{forloop}=="{loop_name}" and eval_type=={filter_eval}')
                # train_on = np.unique(dataframe_filter['train_on'])[0]

                plt.figure(figsize=(12, 8))
                sns_plot = sns.barplot(x=x, y=y, hue=hue, data=dataframe_filter)
                plt.xlabel(x, fontsize=20),
                plt.ylabel('R', fontsize=20)
                plt.title(f'{loop_name}', fontsize=20);
                plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
                # plt.ylim(bottom=.7, top=1.0)

                plt.show()
                # save plot
                figure = sns_plot.get_figure()
                figure.savefig(os.path.join(self.dirs.FIGURES, f'bar-{loop_name}.png'))
        else:
            # filter dataframe for eval
            dataframe_filter = dataframe.query(f'eval_type=={filter_eval}')
            
            plt.figure(figsize=(12, 8))
            sns_plot = sns.barplot(x=x, y=y, hue=hue, data=dataframe_filter)
            plt.xlabel(x, fontsize=20),
            plt.ylabel('R', fontsize=20)
            plt.title(f'{self.eval_name[0]}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
            # plt.ylim(bottom=.7, top=1.0)

            plt.show()

            eval_name = self.eval_name[0]
            figure = sns_plot.get_figure()
            figure.savefig(os.path.join(self.dirs.FIGURES, f'bar-{eval_name}-{x}-{filter_eval}-{hue}.png'))

        # optionally plot model and eval params
        if plot_params:
            # pprint.pprint(self._get_config_params(), compact=True)
            pprint.pprint(self.config, compact=True)
   
    def _get_config_params(self):
        # load params from trained models
        defaults = Defaults()

        # read in config file
        # config = io.read_json(str(defaults.model_config))
        
        # update config file with inputs
        config.update(self.config)

        return config

class MapCerebellum(Utils):
    """ Map Visualization Class for Cerebellum: converts voxel numpy arrays
        from model evaluation to nifti and gifti format
        visualizes gifti files on flatmap surface of the cerebellum
    """

    def __init__(self, config, **kwargs):
        """ 
            Args: 
                config (dict): dictionary loaded from `visualize_cerebellum_config.json` containing 
                parameters for visualizing cerebellar surface maps

            Kwargs:
                eval_name (str): eval name default is 'tesselsWB642_grey_nan_l2_regress'
                subjects (list of int): list of subjects. see constants.py for subject list
                data_type (str): default is "S_best_weight". other options are "R_pred", "R_y", "R_pred_crossed", 'R_pred_uncrossed", "S_ginni"
                mask_name (str): default is "cerebellarGreySUIT.nii"
                surf_mesh (str): default is "FLAT.surf.gii"
                eval_on (list of str): study(s) to be used for evaluation. default is ['sc1', 'sc2']
                glm (int):  default is 7. options are 7 and 8
                surface_threshold (int): default is null
                symmetric_cmap (bool): default is False
                colorbar (bool): default is True
                view (str): option for viewing surface data. default is 'resize'. other option is 'browser'
        """
        self.config = copy.deepcopy(config)
        self.config.update(**kwargs)

    def visualize_subj(self):

        # save files to nifti first
        self.save_data_to_nifti()

        for exp in self.config['eval_on']:

            # set directories for exp
            self.dirs = Dirs(study_name=exp, glm=self.config['glm'])

            # get all model dirs for `eval_name`
            eval_name = self.config['eval_name']
            eval_dirs = self.get_all_files(fullpath=self.dirs.SUIT_GLM_DIR, wildcard=f'*{eval_name}*')

            # loop over models
            for eval_dir in eval_dirs:
                
                # loop over subjects
                for subj in self.config['subjects']:

                    # get gifti files for `eval_name`, `subj`, `exp`, `data_type`
                    data_type = self.config['data_type']
                    gifti_fnames = self.get_all_files(fullpath=os.path.join(eval_dir, f's{subj:02}'), wildcard=f'*{data_type}_vox.gii*')
            
                    # loop over gifti files
                    for gifti_fpath in gifti_fnames:

                        # plot gifti if it exists
                        if os.path.exists(gifti_fpath):

                            # print(io.read_json(fpath=os.path.join(self.dirs.CONN_EVAL_DIR, Path(eval_dir).name + '.json')))

                            # plot map on surface
                            self._plot_surface_cerebellum(surf_map=surface.load_surf_data(gifti_fpath),
                                                    surf_mesh=os.path.join(self.dirs.ATLAS_SUIT_FLATMAP, self.config['surf_mesh']),
                                                    title=f'{exp}:{Path(gifti_fpath).stem}') 
                        else:
                            print(f'no gifti file for {data_type} for {exp}')

    def visualize_group(self):

        # save files to nifti first
        self.save_data_to_nifti()

        for exp in self.config['eval_on']:

            # set directories for exp
            self.dirs = Dirs(study_name=exp, glm=self.config['glm'])

            # get all model dirs for `eval_name`
            eval_name = self.config['eval_name']
            eval_dirs = self.get_all_files(fullpath=self.dirs.SUIT_GLM_DIR, wildcard=f'*{eval_name}*')

            # loop over models and visualize group
            for eval_dir in eval_dirs:

                # get gifti files for `eval_name`, `subj`, `exp`, `data_type`
                data_type = self.config['data_type']
                gifti_fpath = self.get_all_files(fullpath=os.path.join(eval_dir, 'group'), wildcard=f'*{data_type}_vox.gii*')[0]

                if os.path.exists(gifti_fpath):

                    # print out param file to accompany gifti file
                    pprint.pprint(io.read_json(fpath=os.path.join(self.dirs.CONN_EVAL_DIR, Path(eval_dir).name + '.json')), compact=True)

                    # plot group map on surface
                    self._plot_surface_cerebellum(surf_map=surface.load_surf_data(gifti_fpath),
                                            surf_mesh=os.path.join(self.dirs.ATLAS_SUIT_FLATMAP, self.config['surf_mesh']),
                                            title=f'{data_type} : eval on {exp} : {Path(eval_dir).name}') 
                else:
                    print(f'no gifti file for {data_type} for {exp}')

    def save_data_to_nifti(self):
        """ saves predictions to nifti files
            niftis need to be mapped to surf, this is done in matlab (using suit)
            matlab engine will eventually be called from python to bypass this step
            *** not yet implemented ***
        """
        # loop over exp
        for exp in self.config['eval_on']:

            self.dirs = Dirs(study_name=exp, glm=self.config['glm'])

            # get fnames for eval data for `eval_name` and for `exp`
            eval_name = self.config['eval_name']
            fnames = self.get_all_files(fullpath=self.dirs.CONN_EVAL_DIR, wildcard=f'*{eval_name}*.h5')

            # save individual subj voxel predictions
            # and group avg. to nifti files
            self._convert_to_nifti(files=fnames)

    def _convert_to_nifti(self, files):
        """ converts outputs from `files` to nifti
        """
        # loop over file names for `eval_name`
        for self.file in files:

            # load prediction dict for `eval_name` and `data_type`
            data_dict = self._load_data()

            data_type = self.config['data_type']
            data_dict = {k:v for k,v in data_dict.items() if f'{data_type}_vox' in k} 

            # only convert vox data to nifti
            if data_dict:

                print(f'{self.file} contains vox data')

                # loop over all prediction data
                nib_objs = []
                for self.name in data_dict:

                    # get outpath to niftis
                    nifti_subj_fpath, nifti_group_fpath = self._get_nifti_outpath()

                    # convert subj to nifti if it doesn't already exist
                    if not os.path.exists(nifti_subj_fpath):

                        # get input data for nifti obj
                        Y, non_zero_ind, mask = self._get_nifti_input_data(data_dict=data_dict)

                        # get vox data as nib obj
                        nib_obj = image_utils.convert_to_vol(Y=Y, vox_indx=non_zero_ind, mask=mask)
                        nib_objs.append(nib_obj)

                        # save nifti obj to disk
                        image_utils.save_nifti_obj(nib_obj, nifti_subj_fpath)
                        print(f'saved {nifti_subj_fpath} to nifti, please map this file surf in matlab')

                # convert group to nifti if it doesn't already exist
                if not os.path.exists(nifti_group_fpath):
                    # calculate group avg nifti of `data_type` for `eval_name` 
                    image_utils.calc_nifti_average(imgs=nib_objs, fpath=nifti_group_fpath)
                    print(f'saved {nifti_group_fpath} to nifti, please map this file surf in matlab')
            
            else:
                print(f'{self.file} does not have vox data')

    def _get_nifti_outpath(self):
        """ returns nifti fpath for subj and group prediction maps
        """
        self.subj_name = re.findall('(s\d+)', self.name)[0] # extract subj name
        nifti_fname = f'{self.name}.nii' # set nifti fname

        # get model, subj, group dirs in suit
        SUIT_MODEL_DIR = os.path.join(self.dirs.SUIT_GLM_DIR, Path(self.file).stem )
        SUBJ_DIR = os.path.join(SUIT_MODEL_DIR, self.subj_name) # get nifti dir for subj
        GROUP_DIR = os.path.join(SUIT_MODEL_DIR, 'group') # get nifti dir for group

        # make subj and group dirs in suit if they don't already exist
        for _dir in [SUBJ_DIR, GROUP_DIR]:
            self._make_dir(_dir)

        # get full path to nifti fname for subj and group
        subj_fpath = os.path.join(SUBJ_DIR, nifti_fname)
        group_fpath = os.path.join(GROUP_DIR, re.sub('(s\d+)', 'group', nifti_fname))

        # return fpath to subj and group nifti
        return subj_fpath, group_fpath

    def _get_nifti_input_data(self, data_dict):
        """ get mask, voxel_data, and vox indices to
            be used as input for `make_nifti_obj`
            Args: 
                data_dict (dict): data dict containing voxel data (numpy array)
            Returns:
                mask (nib obj), Y (numpy array), non_zero_ind (numpy array)
        """
        # get prediction data for `name`
        Y = data_dict[self.name][0]

        # load in `grey_nan` indices
        non_zero_ind_dict = io.read_json(os.path.join(self.dirs.ENCODE_DIR, 'grey_nan_nonZeroInd.json'))
        non_zero_ind = [int(vox) for vox in non_zero_ind_dict[self.subj_name]] 

        # get cerebellar mask
        mask = self._get_cerebellar_mask(mask=self.config['mask_name'], glm=self.config['glm'])

        return Y, non_zero_ind, mask
    
    def _load_data(self):
        data_dict_all = {}
        data_dict_all['all-keys'] = self._load_data_file(data_fname=self.file)

        # conjoin nested keys (will form nifti filenames)
        return self._flatten_nested_dict(data_dict_all) 

    def _plot_surface_cerebellum(self, surf_map, surf_mesh, title):
        """ plots data to flatmap, opens in browser (default)
            Args: 
                surf_map (numpy array): data to plot
                surf_mesh (numpy array): mesh surface. default is "FLAT_SURF"
                title (str): title of plot
        """
        view = plotting.view_surf(surf_mesh=surf_mesh, 
                                surf_map=surf_map, 
                                colorbar=self.config['colorbar'],
                                threshold=self.config['surface_threshold'],
                                vmax=max(surf_map),
                                vmin=min(surf_map),
                                symmetric_cmap=self.config['symmetric_cmap'],
                                title=title) 

        if self.config['view'] == 'browser':
            view.open_in_browser()
        else: 
            view.resize(500, 500)

class MapCortex(Utils):
    """ Map Visualization Class for Cortex:
        visualizes gifti files on the cortex
    """

    def __init__(self, config, **kwargs):
        """ 
            Args: 
                config (dict): dictionary loaded from `visualize_cortex_config.json` containing 
                parameters for visualizing cerebellar surface maps

            Kwargs:
                model_name (str): model name default is 'tesselsWB642_grey_nan_l2_regress'
                subjects (list of int): list of subjects. see constants.py for subject list
                data_type (str): default is "weights". other options are ....
                mask_name (str): default is "cerebellarGreySUIT.nii"
                surf_mesh (str): default is "FLAT.surf.gii"
                train_on (list of str): study(s) to be used for training. default is ['sc1', 'sc2']
                glm (int):  default is 7. options are 7 and 8
                surface_threshold (int): default is null
                colorbar (bool): default is True
                view (str): option for viewing surface data. default is 'resize'. other option is 'browser'
                vmax (int): default is 30
                interactive_threshold (str): default is '10%'
                display_mode (str): default is 'ortho'
                plot_abs (bool): default is False
                interactive (bool): False
                hemi (list of str): default is ['right', 'left']
        """
        self.config = copy.deepcopy(config)
        self.config.update(**kwargs)

    def plot_glass_brain(self):
        plotting.plot_glass_brain(self.img, colorbar=self.config['colorbar'], 
                                threshold=self.config['surface_threshold'],
                                title=self.title, plot_abs=self.config['plot_abs'], 
                                display_mode=self.config['display_mode'])
        plt.show()

    def plot_surface(self): 
        plotting.plot_surf_stat_map(self.inflated, self.texture,
                                    hemi=self.hem, threshold=self.config['surface_threshold'],
                                    title=self.config['title'], colorbar=self.config['colorbar'], 
                                    bg_map=self.config['bg_map'], vmax=self.config['vmax'])
        
        plt.show()

    def plot_interactive_surface(self):
        view = plotting.view_surf(self.inflated, self.texture, 
                                threshold=self.interactive_threshold, 
                                bg_map=self.bg_map)

        view.open_in_browser()

    def _get_surfaces(self):
        if self.hem=="right":
            inflated = self.fsaverage.infl_right
            bg_map = self.fsaverage.sulc_right
            texture = surface.vol_to_surf(self.img, self.fsaverage.pial_right)
        elif self.hem=="left":
            inflated = self.fsaverage.infl_left
            bg_map = self.fsaverage.sulc_left
            texture = surface.vol_to_surf(self.img, self.fsaverage.pial_left)
        else:
            print('hemisphere not provided')

        return inflated, bg_map, texture
    
    def visualize_subj_glass_brain(self):

        # loop over subjects
        for subj in Defaults.subjs:

            # GLM_SUBJ_DIR = os.path.join(Defaults.GLM_DIR, self.config['glm'], subj)
            os.chdir(GLM_SUBJ_DIR)
            # fpaths = glob.glob(f'*{self.contrast_type}-{self.glm}.nii')
            
            for fpath in fpaths: 
                self.img = nib.load(os.path.join(GLM_SUBJ_DIR, fpath))
                self.title = f'{Path(fpath).stem})'
                self.plot_glass_brain()

    def visualize_group_glass_brain(self):

        # GLM_GROUP_DIR = os.path.join(Defaults.GLM_DIR, self.glm, "group")
        os.chdir(GLM_GROUP_DIR)
        fpaths = glob.glob(f'*{self.contrast_type}-{self.glm}.nii')

        for fpath in fpaths:
            # define contrast names
            self.img = nib.load(os.path.join(GLM_GROUP_DIR, fpath))
            self.title = f'{Path(fpath).stem})'
            self.plot_glass_brain()

    def visualize_group_surface(self):
        self.fsaverage = datasets.fetch_surf_fsaverage()

        # GLM_GROUP_DIR = os.path.join(Defaults.GLM_DIR, self.glm, "group")
        os.chdir(GLM_GROUP_DIR)
        # fpaths = glob.glob(f'*{self.contrast_type}-{self.glm}.nii')

        for fpath in fpaths:
            self.img = nib.load(os.path.join(GLM_GROUP_DIR, fpath))

            # loop over hemispheres and visualize
            for self.hem in self.hemi:

                # set plot title
                self.title = f'{Path(fpath).stem}-{self.hem}'

                # return surfaces and texture
                self.inflated, self.bg_map, self.texture = self._get_surfaces()

                # plot interactive or static surface(s)
                if self.interactive:  
                    self.plot_interactive_surface()
                else:
                    self.plot_surface()

        return self.fsaverage

class PlotBetas(DataManager):
    
    def __init__(self):
        super().__init__()
        self.inputs = {'X': {'roi': 'tesselsWB162', 'file_dir': 'encoding', 'structure': 'cortex'},
                       'Y': {'roi': 'grey_nan', 'file_dir': 'encoding', 'structure': 'cerebellum'}}
        self.scale = True
        self.defaults = Defaults()
        self.subjects = self.defaults.return_subjs
        self.dirs = Dirs()

    def load_dataframe(self):
        fpath = os.path.join(self.dirs.BASE_DIR, 'task_betas.csv')
        
        # if avg betas file doesn't exist, save to disk
        if not os.path.exists(fpath):
            self._save_to_disk()

        return pd.read_csv(fpath)

    def _save_to_disk(self):
        data ={}
        for roi in self.inputs:

            self.data_type = self.inputs[roi]
            data[roi] = self.get_conn_data()

        # get avg betas for cortex and cerebellum
        roi_dict = self._calc_avg_betas(data_dict=data)

        # horizontally concat both dataframes
        X = pd.DataFrame.from_dict(roi_dict['X'])
        Y = pd.DataFrame.from_dict(roi_dict['Y'])
        dataframe = pd.concat([X, Y], axis=1)

        # drop duplicate columns
        dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()] # drop duplicate cols

        # save avg betas to disk
        dataframe.to_csv(os.path.join(self.dirs.BASE_DIR, 'task_betas.csv'), index=False)

    def _calc_avg_betas(self, data_dict):

        # loop over rois
        roi_dict = {}
        for roi in data_dict:

            structure = self.inputs[roi]['structure']
            roi_name = self.inputs[roi]['roi']
            
            avg_betas = defaultdict(list)
            for subj in self.subjects:
                print(f'averaging betas for s{subj:02}...')

                # get subj betas
                subj_data = data_dict[roi]['betas'][f's{subj:02}']
                if self.scale:
                    subj_data = self._scale_data(subj_data)

                # get cond/task names
                stim_nums = np.array(data_dict[roi][f'{self.stim}Num'])
                stim_names = np.array(data_dict[roi][f'{self.stim}Name'])

                # loop over exp and sess
                for exp, sess in zip(np.arange(1,len(self.experiment)+1), self.sessions):

                    # get index for `exp` and `sess`
                    exp_idx = np.array(data_dict[roi]['StudyNum']) == exp
                    sess_idx = np.array(data_dict[roi]['sess']) == sess
                    idx = [a and b for a, b in zip(exp_idx, sess_idx)]     

                    # loop over `stim`
                    for num, name in zip(stim_nums[idx,], stim_names[idx,]):

                        # get avg beta
                        avg_beta = np.nanmean(subj_data[idx,][num,])

                        # append data to dict
                        data = {'study': exp, 'sess': sess,
                                'stim_num': num, 'stim_name': name,
                                f'{structure}_beta': avg_beta, 'subj': subj,
                                'roi': roi_name,
                                'structure': structure}

                        for k,v in data.items():
                            avg_betas[k].append(v)

            roi_dict[roi] = avg_betas

        return roi_dict

    def task_scatter_all(self, dataframe):

        dataframe = dataframe[['stim_name', 'cortex_beta', 'cerebellum_beta']].groupby('stim_name').agg('mean').reset_index()

        sns.set(rc={'figure.figsize':(10,10)})
        sns.scatterplot(x="cerebellum_beta", y="cortex_beta", data=dataframe)

        # plot regression line
        m, b = np.polyfit(dataframe['cerebellum_beta'], dataframe['cortex_beta'], 1)
        plt.plot(dataframe['cerebellum_beta'], m*dataframe['cerebellum_beta']+ b)

        def label_point(x, y, val, ax):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                ax.text(point['x']+.02, point['y'], str(point['val']), fontsize=10)

        label_point(dataframe['cerebellum_beta'], dataframe['cortex_beta'], dataframe['stim_name'], plt.gca())

        plt.xlabel('cerebellum', fontsize=20),
        plt.ylabel('cortex', fontsize=20)
        plt.title('avg. betas: study 1 & study 2', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        # plt.ylim(bottom=.7, top=1.0)

        plt.show()

    def task_scatter_interactive_all(self, dataframe):
        dataframe = dataframe[['stim_name', 'cortex_beta', 'cerebellum_beta']].groupby('stim_name').agg('mean').reset_index()

        fig = go.Figure(data=go.Scatter(x=dataframe['cerebellum_beta'],
                        y=dataframe['cortex_beta'],
                        mode='markers',
                        marker=dict(
                        color='LightSkyBlue',
                        size=20,
                        line=dict(
                        color='MediumPurple',
                        width=2
                        )),
                        text=dataframe['stim_name'], 
                        textfont=dict(
                        family="sans serif",
                        size=18,
                        color="LightSeaGreen")))

        # plot regression line
        m, b = np.polyfit(dataframe['cortex_beta'], dataframe['cerebellum_beta'], 1)

        fig.add_trace(go.Scatter(
                x=dataframe['cortex_beta'], y=m*dataframe['cortex_beta']+ b,
                name='regression line',
                marker_color='black'
                ))

        fig.update_layout(
            height=800,
            title_text=f'Avg. betas: study 1 & study 2'
            )

        fig.update_xaxes(
            title_font=dict(size=18),
            title='Cerebellum',
            )

        fig.update_yaxes(
            title_font=dict(size=18),
            title='Cortex',
            )

        fig.show()

    def task_scatter_study(self, dataframe):

        for exp in [1, 2]:
            dataframe_exp = dataframe.query(f'study=={exp}')

            # average across task conditions
            dataframe_exp = dataframe_exp[['stim_name', 'cortex_beta', 'cerebellum_beta']].groupby('stim_name').agg('mean').reset_index()

            sns.set(rc={'figure.figsize':(10,10)})
            sns.scatterplot(x="cerebellum_beta", y="cortex_beta", data=dataframe_exp)

            # plot regression line
            m, b = np.polyfit(dataframe_exp['cerebellum_beta'], dataframe_exp['cortex_beta'], 1)
            plt.plot(dataframe_exp['cerebellum_beta'], m*dataframe_exp['cerebellum_beta']+ b)

            def label_point(x, y, val, ax):
                a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
                for i, point in a.iterrows():
                    ax.text(point['x']+.02, point['y'], str(point['val']), fontsize=10)

            label_point(dataframe_exp['cerebellum_beta'], dataframe_exp['cortex_beta'], dataframe_exp['stim_name'], plt.gca())

            plt.xlabel('cerebellum', fontsize=20),
            plt.ylabel('cortex', fontsize=20)
            plt.title(f'avg betas: study {exp}', fontsize=20);
            plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
            # plt.ylim(bottom=.7, top=1.0)

            plt.show()

    def task_scatter_interactive_study(self, dataframe):
        
        for exp in [1, 2]:

            # get study 
            dataframe_exp = dataframe.query(f'study=={exp}')
        
            # avg. across tasks 
            dataframe_exp = dataframe_exp[['stim_name', 'cortex_beta', 'cerebellum_beta']].groupby('stim_name').agg('mean').reset_index()

            fig = go.Figure(data=go.Scatter(x=dataframe_exp['cerebellum_beta'],
                            y=dataframe_exp['cortex_beta'],
                            mode='markers',
                            marker=dict(
                            color='LightSkyBlue',
                            size=20,
                            line=dict(
                            color='MediumPurple',
                            width=2
                            )),
                            text=dataframe_exp['stim_name'], 
                            textfont=dict(
                            family="sans serif",
                            size=18,
                            color="LightSeaGreen")))

            # plot regression line
            m, b = np.polyfit(dataframe_exp['cortex_beta'], dataframe_exp['cerebellum_beta'], 1)

            fig.add_trace(go.Scatter(
                    x=dataframe_exp['cortex_beta'], y=m*dataframe_exp['cortex_beta']+ b,
                    name='regression line',
                    marker_color='black'
                    ))

            fig.update_layout(
                height=800,
                title_text=f'Avg. betas: study {exp}'
                )

            fig.update_xaxes(
                title_font=dict(size=18),
                title='Cerebellum',
                )

            fig.update_yaxes(
                title_font=dict(size=18),
                title='Cortex',
                )

            fig.show()
