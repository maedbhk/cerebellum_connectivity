from pathlib import Path
import os

"""
Created on Aug 10 09:10:51 2020
Defaults for running and visualizing connectivity models
Defaults just contains simple variables, so these should be made module-wide global variables. They can be change from outside with global keyoard as constants.return_subjects 
constants.base_dir

Dirs can remain a class (although its a bit overkill and I probably wouldn't have chosen to do so).
"""

# Module-wide global variables - there is no need to package them into a class! 
return_subjs = [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
# Set the local path here... 
# When committing, leave other people's path in here. 
base_dir = Path('/Volumes/diedrichsen_data$/data/super_cerebellum')

class Dirs: 
    def __init__(self, study_name='sc1', glm=7):
        self.base_dir = base_dir
        self.data_dir = base_dir / study_name
        self.behav_dir = self.data_dir / 'data'
        self.imaging_dir = self.data_dir / 'imaging_data'
        self.suit_dir = self.data_dir / 'suit'
        self.suit_glm_dir = self.suit_dir / f'glm{glm}'
        self.suit_anat_dir = self.suit_dir / 'anatomicals'
        self.reg_dir = self.data_dir / 'RegionOfInterest'
        self.glm_dir = self.data_dir / f'GLM_firstlevel_{glm}'
        self.beta_reg_dir = self.data_dir / 'beta_roi' / f'glm{glm}'
        self.conn_train_dir = self.data_dir / 'conn_models' / 'train'
        self.conn_eval_dir = self.data_dir / 'conn_models' / 'eval'
        self.atlas = base_dir / 'atlases'
        self.atlas_suit_flatmap = self.atlas / 'suit_flatmap'
        self.figure = Path(__file__).absolute().parent.parent / 'reports' / 'figures'

        # create folders if they don't already exist
        # fpaths = [self.BETA_REG_DIR, self.CONN_TRAIN_DIR, self.CONN_EVAL_DIR, self.ATLAS]
        # for fpath in fpaths:
        #     if not os.path.exists(fpath):
        #         print(f'{fpath} should already exist, check your folder transfer!')
        # os.makedirs(fpath)