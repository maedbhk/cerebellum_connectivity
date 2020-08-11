from pathlib import Path
import os

class Defaults: 

    def __init__(self, study_name, glm):
        self.BASE_DIR = Path(__file__).absolute().parent.parent / 'data'
        self.DATA_DIR = self.BASE_DIR / study_name
        self.BEHAV_DIR = self.DATA_DIR / 'data'
        self.IMAGING_DIR = self.DATA_DIR / 'imaging_data'
        self.SUIT_DIR = self.DATA_DIR / 'suit'
        self.REG_DIR = self.DATA_DIR / 'RegionOfInterest'
        self.GLM_DIR = self.DATA_DIR / f'GLM_firstlevel_{glm}'
        self.ENCODE_DIR = self.DATA_DIR / 'encoding'
        self.ENCODE_GLM_DIR = self.ENCODE_DIR / f'glm{glm}'
        self.BETA_REG_DIR = self.DATA_DIR / 'beta_roi'
        self.BETA_REG_GLM_DIR = self.BETA_REG_DIR / f'glm{glm}'
        self.CONN_DIR = self.DATA_DIR / 'conn_models'

        # self.return_subjs = [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
        self.return_subjs = [3, 4]

        self.conn_file = 'sc1_sc2_taskConds_conn.txt'

        # create folders if they don't already exist
        fpaths = [self.BETA_REG_DIR, self.BETA_REG_GLM_DIR]
        for fpath in fpaths:
            if not os.path.exists(fpath):
                print(f'creating {fpath} although this dir should already exist, check your folder transfer!')
                os.makedirs(fpath)