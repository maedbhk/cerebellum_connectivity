import os
import glob

from connectivity import io
from connectivity.models.evaluate_model import EvaluateModel
from connectivity.models.train_model import TrainModel
from connectivity.constants import Defaults, Dirs

def delete_conn_files():
    # delete any pre-existing connectivity files
    for exp in ['sc1', 'sc2']:
        dirs = Dirs(study_name=exp, glm=7)
        filelist = glob.glob(os.path.join(dirs.CONN_TRAIN_DIR, '*'))
        for f in filelist:
            os.remove(f)

def get_config_file():
    # get dirs
    dirs = Dirs()

    # load config files for train and eval parameters
    return io.read_json(os.path.join(dirs.BASE_DIR, 'config.json'))

def train(config, **kwargs):
    """ This routine does model training and model evaluation
        Args: 
            config_train (dict): dictionary loaded from `config_train.json`
            
            Kwargs:
                model_name (str): model name default is "l2_regress"
                sessions (list): default is [1, 2]. options are [1, 2]
                glm (int):  default is 7. options are 7 and 8
                stim (str): default is 'cond'. options are 'cond' and 'task' (depends on glm)
                avg (str): average over 'run' or 'sess'. default is 'run'
                incl_inst (bool): default is True. 
                subtract_sess_mean (bool): default is True.
                subtract_exp_mean (bool): default is True.
                subjects (list of int): list of subjects. see constants.py for subject list. 
                train_on (str): study to be used for training. default is 'sc1'. options are 'sc1' or 'sc2'.
                train_inputs (nested dict): primary keys are `X` and `Y`. secondary keys are 'roi', 'file_dir', 'structure'
                train_mode (str): training mode: 'crossed' or 'uncrossed'. If 'crossed': sessions are flipped between `X` and `Y`. default is 'crossed'
                scale (bool): normalize `X` and `Y` data. default is True.
                lambdas (list of int): list of lambdas if `model_name` = 'l2_regress'
                n_pcs (list of int): list of pcs if `model_name` = 'pls_regress'
    """
    
    # train
    model = TrainModel(config, **kwargs)
    model.model_train() 

def evaluate(config, **kwargs):
    """ This routine does model evaluation
        Args: 
            config_eval (dict): dictionary loaded from `config_eval.json`
            
            Kwargs:
                model (str): model name default is 'l2_regress'
                train_sessions (list): Default is [1, 2]. Options are [1, 2]
                train_glm (int):  Default is 7. Options are 7 and 8
                train_stim (str): Default is 'cond'. Options are 'cond' and 'task' (depends on glm)
                train_avg (str): average over 'run' or 'sess'. Default is 'run'
                train_incl_inst (bool): Default is True. 
                train_subtract_sess_mean (bool): Default is True.
                train_subtract_exp_mean (bool): Default is True.
                train_subjects (list of int): list of subjects. see constants.py for subject list. 
                train_on (str): study to be used for training. Default is 'sc1'. Options are 'sc1' or 'sc2'.
                train_inputs (nested dict): primary keys are `X` and `Y`. secondary keys are 'roi', 'file_dir', 'structure'
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
                eval_inputs (nested dict): primary keys are `X` and `Y`. secondary keys are 'roi', 'file_dir', 'structure'
                eval_scale (bool): normalize `X` and `Y` data. Default is True.
                eval_splitby (str): split evaluation by 'cond' or 'task' or None. Default is None.
                eval_save_maps (bool): save out predictions and reliabilities of voxel maps. Default is False.
    """
    # evaluate
    model_eval = EvaluateModel(config=config, **kwargs) 
    model_eval.model_evaluate() 

# delete existing connectivity files
delete_conn_files()

# get config files for training and evaluating connectivity data
config_obj = get_config_file()

# train model
train(config=config_obj, model_name='l2_regress', lambdas=[1, 2])

# evaluate model
evaluate(config=config_obj, model_name='l2_regress', eval_splitby='cond', lambdas=[1, 2])