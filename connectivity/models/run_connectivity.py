import os
import sys
import glob
import numpy as np

import click

from connectivity import io
from connectivity.models.evaluate_model import EvaluateModel
from connectivity.models.train_model import TrainModel

from connectivity.constants import Defaults, Dirs

import warnings
warnings.filterwarnings('ignore')

np.seterr(divide='ignore', invalid='ignore')

"""
Created on Aug 31 11:14:19 2020
Main script for training and evaluating connectivity models

@authors: Maedbh King
"""

def _delete_conn_files():
    """ delete any pre-existing connectivity output
    """
    for exp in ['sc1', 'sc2']:
        dirs = Dirs(study_name=exp, glm=7)
        filelists = [glob.glob(os.path.join(dirs.CONN_TRAIN_DIR, '*')), glob.glob(os.path.join(dirs.CONN_EVAL_DIR, '*'))]
        for filelist in filelists:
            for f in filelist:
                os.remove(f)
    print('deleting training and evaluation data')

def _get_config_file():
    # define dirs class
    defaults = Defaults()

    # load config files for train and eval parameters
    return io.read_json(str(defaults.model_config))

def _train_model(config, **kwargs):
    """ This routine does model training and model evaluation
        Args: 
            config (dict): dictionary loaded from `model_config.json`
            
        Kwargs:
            model_name (str): model name default is "l2_regress"
            train_sessions (list): default is [1, 2]. options are [1, 2]
            train_glm (int):  default is 7. options are 7 and 8
            train_stim (str): default is 'cond'. options are 'cond' and 'task' (depends on glm)
            train_avg (str): average over 'run' or 'sess'. default is 'run'
            train_incl_inst (bool): default is True. 
            train_subtract_sess_mean (bool): default is True.
            train_subtract_exp_mean (bool): default is True.
            train_subjects (list of int): list of subjects. see constants.py for subject list. 
            train_on (str): study to be used for training. default is 'sc1'. options are 'sc1' or 'sc2'.
            train_X_roi: 'tesselsWB362'
            train_X_file_dir: 'encoding''
            train_X_structure: 'cortex'
            train_Y_roi: 'grey_nan'
            train_Y_file_dir: 'encoding'
            train_Y_structure: 'cerebellum'
            train_mode (str): training mode: 'crossed' or 'uncrossed'. If 'crossed': sessions are flipped between `X` and `Y`. default is 'crossed'
            train_scale (bool): normalize `X` and `Y` data. default is True.
            lambdas (list of int): list of lambdas if `model_name` = 'l2_regress'
            n_pcs (list of int): list of pcs if `model_name` = 'pls_regress' # not yet implemented
    """
    
    # train
    model = TrainModel(config, **kwargs)
    model.model_train() 

def _evaluate_model(config, **kwargs):
    """ This routine does model evaluation
        Args: 
            config (dict): dictionary loaded from `model_config.json`
            
        Kwargs:
            model_name (str): model name default is 'l2_regress'
            train_sessions (list): Default is [1, 2]. Options are [1, 2]
            train_glm (int):  Default is 7. Options are 7 and 8
            train_stim (str): Default is 'cond'. Options are 'cond' and 'task' (depends on glm)
            train_avg (str): average over 'run' or 'sess'. Default is 'run'
            train_incl_inst (bool): Default is True. 
            train_subtract_sess_mean (bool): Default is True.
            train_subtract_exp_mean (bool): Default is True.
            train_subjects (list of int): list of subjects. see constants.py for subject list. 
            train_on (str): study to be used for training. Default is 'sc1'. Options are 'sc1' or 'sc2'.
            train_X_roi (str): 'tesselsWB362'
            train_X_file_dir (str): 'encoding''
            train_X_structure (str): 'cortex'
            train_Y_roi (str): 'grey_nan'
            train_Y_file_dir (str): 'encoding'
            train_Y_structure (str): 'cerebellum'
            train_mode (str): training mode: 'crossed' or 'uncrossed'. If 'crossed': sessions are flipped between `X` and `Y`. Default is 'crossed'
            train_scale (bool): normalize `X` and `Y` data. Default is True.

            eval_sessions (list): Default is [1, 2]. Options are [1, 2]
            eval_glm (int):  Default is 7. Options are 7 and 8
            eval_stim (str): Default is 'cond'. Options are 'cond' and 'task' (depends on glm)
            eval_avg (str): average over 'run' or 'sess'. Default is 'run'
            eval_incl_inst (bool): Default is True. 
            eval_subtract_sess_mean (bool): Default is True.
            eval_subtract_exp_mean (bool): Default is True.
            eval_subjects(list of int): list of subjects. see constants.py for subject list. 
            eval_on (str): study to be used for training. Default is 'sc2'. Options are 'sc1' or 'sc2'.
            eval_X_roi (str): 'tesselsWB362'
            eval_X_file_dir (str): 'encoding''
            eval_X_structure (str): 'cortex'
            eval_Y_roi (str): 'grey_nan'
            eval_Y_file_dir (str): 'encoding'
            eval_Y_structure (str): 'cerebellum'
            eval_scale (bool): normalize `X` and `Y` data. Default is True.
            eval_splitby (str): split evaluation by 'cond' or 'task' or None. Default is None.
            eval_save_maps (bool): save out predictions and reliabilities of voxel maps. Default is False.
            lambdas (list of int): list of lambdas if `model_name` = 'l2_regress'
            n_pcs (list of int): list of pcs if `model_name` = 'pls_regress' # not yet implemented
    """
    # evaluate
    model_eval = EvaluateModel(config=config, **kwargs) 
    model_eval.model_evaluate() 

def train_evaluate(train=True, evaluate=True, **kwargs):
    """ This routine does model training and evaluation
        Args: 
            config (dict): dictionary loaded from `model_config.json`
            train (bool): default is True
            evaluate (bool): default is True
            
        Kwargs:
            model_name (str): model name default is 'l2_regress'
            train_sessions (list): Default is [1, 2]. Options are [1, 2]
            train_glm (int):  Default is 7. Options are 7 and 8
            train_stim (str): Default is 'cond'. Options are 'cond' and 'task' (depends on glm)
            train_avg (str): average over 'run' or 'sess'. Default is 'run'
            train_incl_inst (bool): Default is True. 
            train_subtract_sess_mean (bool): Default is True.
            train_subtract_exp_mean (bool): Default is True.
            train_subjects (list of int): list of subjects. see constants.py for subject list. 
            train_on (str): study to be used for training. Default is 'sc1'. Options are 'sc1' or 'sc2'.
            train_X_roi (str): 'tesselsWB362'
            train_X_file_dir (str): 'encoding''
            train_X_structure (str): 'cortex'
            train_Y_roi (str): 'grey_nan'
            train_Y_file_dir (str): 'encoding'
            train_Y_structure (str): 'cerebellum'
            train_mode (str): training mode: 'crossed' or 'uncrossed'. If 'crossed': sessions are flipped between `X` and `Y`. Default is 'crossed'
            train_scale (bool): normalize `X` and `Y` data. Default is True.

            eval_sessions (list): Default is [1, 2]. Options are [1, 2]
            eval_glm (int):  Default is 7. Options are 7 and 8
            eval_stim (str): Default is 'cond'. Options are 'cond' and 'task' (depends on glm)
            eval_avg (str): average over 'run' or 'sess'. Default is 'run'
            eval_incl_inst (bool): Default is True. 
            eval_subtract_sess_mean (bool): Default is True.
            eval_subtract_exp_mean (bool): Default is True.
            eval_subjects(list of int): list of subjects. see constants.py for subject list. 
            eval_on (str): study to be used for training. Default is 'sc2'. Options are 'sc1' or 'sc2'.
            eval_X_roi (str): 'tesselsWB362'
            eval_X_file_dir (str): 'encoding''
            eval_X_structure (str): 'cortex'
            eval_Y_roi (str): 'grey_nan'
            eval_Y_file_dir (str): 'encoding'
            eval_Y_structure (str): 'cerebellum'
            eval_scale (bool): normalize `X` and `Y` data. Default is True.
            eval_splitby (str): split evaluation by 'cond' or 'task' or None. Default is None.
            eval_save_maps (bool): save out predictions and reliabilities of voxel maps. Default is False.
            lambdas (list of int): list of lambdas if `model_name` = 'l2_regress'
            n_pcs (list of int): list of pcs if `model_name` = 'pls_regress' # not yet implemented
    """

    # _delete_conn_files()

    # get config files for training and evaluating connectivity data
    config_obj = _get_config_file()
    print('fetching model config')

    # train model
    if train:
        print('training model ...')
        _train_model(config=config_obj, **kwargs)

    # evaluate model
    if evaluate:
        print('evaluating model ...')
        _evaluate_model(config=config_obj, **kwargs)

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--train')
@click.option('--evaluate')
@click.pass_context

def run(ctx, train=True, evaluate=True):
    # this is pretty hacky
    # each arg should really be specified as a click option
    # but it would be verbose given the number of kwargs

    d = dict()
    for item in ctx.args:
        arg = item.split("=")
        key = arg[0]
        val = eval(arg[1])
        d[key] = val

     # input exploded dictionary to `train_evaluate`
    train_evaluate(**d)

if __name__ == '__main__':
    run()

