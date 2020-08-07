from pathlib import Path
import os

class Defaults: 

    def __init__(self, study_name, glm):
        self.BASE_DIR = Path(__file__).absolute().parent.parent
        self.DATA_DIR = self.BASE_DIR / 'data' / study_name
        self.BEHAV_DIR = self.DATA_DIR / 'data'
        self.IMAGING_DIR = self.DATA_DIR / 'imaging_data'
        self.SUIT_DIR = self.DATA_DIR / 'suit'
        self.REG_DIR = self.DATA_DIR / 'RegionOfInterest'
        self.GLM_DIR = self.DATA_DIR / f'GLM_firstlevel_{glm}'
        self.ENCODE_DIR = self.DATA_DIR / 'encoding'
        self.CONN_DIR = self.DATA_DIR / 'conn_models'

        # self.return_subjs = [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
        self.return_subjs = [3]

        # # create folders if they don't already exist
        # fpaths = [self.DATA_DIR, self.BEHAV_DIR, self.IMAGING_DIR, self.SUIT_DIR, self.REG_DIR, self.CONN_DIR, self.GLM_DIR]
        # for fpath in fpaths:
        #     if not os.path.exists(fpath):
        #         print(f'{fpath} should exist but does not!')
        #         if fpath==self.CONN_DIR:
        #             os.makedirs(fpath)