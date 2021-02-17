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

import connectivity.constants as const 
import connectivity.io as cio

import warnings
warnings.filterwarnings('ignore')

"""
Created on Sep 05 07:03:32 2020
Visualization routine for connectivity models

@author: Maedbh King, Jörn Diedrichsen 
"""

def convert_to_vol(Y, vox_indx, mask):
    """
    This function converts 1D numpy array data to 3D vol space, and returns nib obj
    that can then be saved out as a nifti file
    Args:
        Y (numpy array): voxel data, shape (num_vox, )
        vox_indx (numpy array of int): non-zero indices for grey matter voxels in cerebellum, shape (num_vox, )
        mask (nib obj): nib obj of mask
    
    Returns: 
        Nib Obj
    
    """
    # get dat, mat, and dim from the mask
    dat = mask.get_fdata()
    dim = dat.shape
    mat = mask.affine
    
    # initialise xyz voxel data
    vol_data = np.zeros((dim[0], dim[1], dim[2], 1))
    
    # get the xyz of the nonZeroVoxels used in estimating connectivity weights
    (x, y, z)= np.unravel_index(vox_indx, mask.shape, 'F')
    
    num_vox = len(Y)
    for i in range(num_vox):
        vol_data[x[i], y[i], z[i]] = Y[i]

    return make_nifti_obj(vol_data=vol_data, affine_mat=mat)

def calc_nifti_average(imgs, fpath):
    """ calculates average nifti image
        Args: 
            imgs (list): iterable of Niimg-like objects
            out_path (str): full path to averaged image
        Returns: 
            saves averaged nifti file to disk
    """
    mean_img = image.mean_img(imgs) # get average of images from `filenames` list
    # save nifti to disk
    save_nifti_obj(nib_obj=mean_img, fpath=fpath)

def get_cerebellar_mask(self, mask, glm):
    """ converts cerebellar mask to nifti obj
        Args: 
            mask (str): make name
        Returns: 
            nifti obj of mask
    """
    dirs = Dirs(study_name='sc1', glm=glm)
    return nib.load(os.path.join(dirs.SUIT_ANAT_DIR, mask))

def get_all_files(self, fullpath, wildcard):
    return glob.glob(os.path.join(fullpath, wildcard))

def plot_surface_cerebellum(data, surf_mesh = g_surf_mesh, title = None):
    """ plots data to flatmap, opens in browser (default)
        Args: 
            surf_map (numpy array): data to plot
            surf_mesh (numpy array): mesh surface. default is "FLAT_SURF"
            title (str): title of plot
    """
    view = plotting.view_surf(surf_mesh=g_surf_mesh, 
                            surf_map=data, 
                            colorbar=g_colorbar,
                            threshold=g_surface_threshold,
                            vmax=max(data),
                            vmin=min(data),
                            symmetric_cmap=g_symmetric_cmap,
                            title=title) 

    if g_view == 'browser':
        view.open_in_browser()
    else: 
        view.resize(500, 500)

def plot_surface_cortex(self): 
    plotting.plot_surf_stat_map(self.inflated, self.texture,
                                hemi=self.hem, threshold=self.config['surface_threshold'],
                                title=self.config['title'], colorbar=self.config['colorbar'], 
                                bg_map=self.config['bg_map'], vmax=self.config['vmax'])
    
    plt.show()
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


