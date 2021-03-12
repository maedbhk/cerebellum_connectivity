from pathlib import Path
import os
import re


"""Main module for setting defaults for running and visualizing connectivity models
   Defaults contain global variables. They can be changed from outside with global keyboard as constants.return_subjects.

   @authors: Maedbh King, Jörn Diedrichsen

  Typical usage example:

  dirs = Dirs()
"""

return_subjs = ["s02","s03","s04","s06","s08",
        "s09","s10","s12","s14","s15","s17","s18",
        "s19","s20","s21","s22","s24","s25","s26",
        "s27","s28","s29","s30","s31"]

# Set the local path here...
# When committing, leave other people's path in here.
base_dir = Path("/Volumes/diedrichsen_data$/data/super_cerebellum")
base_dir = Path('/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/SuperCerebellum')
#base_dir = Path("global/scratch/maedbhking/projects/cerebellum_connectivity/data")
#base_dir = Path("/Users/maedbhking/Documents/cerebellum_connectivity/data")


class Dirs:
    """Hardcode data directories.

    Args:
        exp_name (str): 'sc1' or 'sc2' experiments
        glm (str): 'glm7', 'glm8'
    """

    def __init__(self, exp_name="sc1", glm="glm7"):
        glm_num = re.findall("\d+", glm)[0]
        self.base_dir = base_dir
        self.data_dir = base_dir / exp_name
        self.behav_dir = self.data_dir / "data"
        self.imaging_dir = self.data_dir / "imaging_data"
        self.suit_dir = self.data_dir / "suit"
        self.suit_glm_dir = self.suit_dir / glm
        self.suit_anat_dir = self.suit_dir / "anatomicals"
        self.reg_dir = self.data_dir / "RegionOfInterest"
        self.glm_dir = self.data_dir / f"GLM_firstlevel_{glm_num}"
        self.beta_reg_dir = self.data_dir / "beta_roi" / glm
        self.conn_train_dir = self.data_dir / "conn_models" / "train"
        self.conn_eval_dir = self.data_dir / "conn_models" / "eval"
        self.atlas = base_dir / "atlases"
        self.atlas_suit_flatmap = self.atlas / "suit_flatmap"
        self.fs_lr_dir = base_dir / "fs_LR_32"
        self.figure = Path(__file__).absolute().parent.parent / "reports" / "figures"
