from pathlib import Path
import os

"""
Created on Aug 10 09:10:51 2020
Constants and defaults for running and visualizing connectivity models

@author: Maedbh King
"""

class Defaults:

    def __init__(self):
        self.return_subjs = [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
        self.task_config =  Path(__file__).absolute().parent.parent  / 'data' / 'task_config.json'
        self.model_config = Path(__file__).absolute().parent.parent  / 'models' / 'model_config.json'
        self.visualize_cerebellum_config = Path(__file__).absolute().parent.parent  / 'visualization' / 'visualize_cerebellum_config.json'
        self.visualize_cortex_config = Path(__file__).absolute().parent.parent  / 'visualization' / 'visualize_cortex_config.json'

class Dirs: 

    def __init__(self, study_name='sc1', glm=7):
        # Set the local path here... 
        # When committing, leave other people's path in here. 
        # self.BASE_DIR = Path(__file__).absolute().parent.parent / 'data'
        self.BASE_DIR = Path('/Volumes/diedrichsen_data$/data/super_cerebellum')
        self.DATA_DIR = self.BASE_DIR / study_name
        self.BEHAV_DIR = self.DATA_DIR / 'data'
        self.IMAGING_DIR = self.DATA_DIR / 'imaging_data'
        self.SUIT_DIR = self.DATA_DIR / 'suit'
        self.SUIT_GLM_DIR = self.SUIT_DIR / f'glm{glm}'
        self.SUIT_ANAT_DIR = self.SUIT_DIR / 'anatomicals'
        self.REG_DIR = self.DATA_DIR / 'RegionOfInterest'
        self.GLM_DIR = self.DATA_DIR / f'GLM_firstlevel_{glm}'
        self.ENCODE_DIR = self.DATA_DIR / 'encoding' / f'glm{glm}'
        self.BETA_REG_DIR = self.DATA_DIR / 'beta_roi' / f'glm{glm}'
        self.CONN_TRAIN_DIR = self.DATA_DIR / 'conn_models' / 'train'
        self.CONN_EVAL_DIR = self.DATA_DIR / 'conn_models' / 'eval'
        self.ATLAS = self.BASE_DIR / 'atlases'
        self.ATLAS_SUIT_FLATMAP = self.ATLAS / 'suit_flatmap'
        self.FIGURES = Path(__file__).absolute().parent.parent / 'reports' / 'figures'

        # create folders if they don't already exist
        fpaths = [self.BETA_REG_DIR, self.CONN_TRAIN_DIR, self.CONN_EVAL_DIR, self.ATLAS]
        # for fpath in fpaths:
        #     if not os.path.exists(fpath):
        #         print(f'{fpath} should already exist, check your folder transfer!')
        # os.makedirs(fpath)