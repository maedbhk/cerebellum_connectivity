import os
import numpy as np
from time import gmtime, strftime  
import glob
import copy
from dictdiffer import diff, patch, swap, revert 

from collections import defaultdict
from functools import partial

from connectivity import io
from connectivity.helper_functions import AutoVivification
from connectivity.data.prep_data import DataManager
from connectivity.data.prep_timeseries_data import DataManager as DataManagerTS
from connectivity.constants import Defaults, Dirs
from connectivity.models.model_functions import MODEL_MAP

import warnings
warnings.filterwarnings('ignore')

np.seterr(divide='ignore', invalid='ignore')

"""
Created on Sat Jun 27 13:34:09 2020
Model evaluation routine for connectivity models

@authors: Ladan Shahshahani and Maedbh King
"""

class EvaluateModel(DataManagerTS):

    def __init__(self, config, **kwargs):
        """ Model evaluation Class, inherits methods from DataManager Class
            Args: 
                config (dict): dictionary loaded from`config.json` containing 
                training and evaluation parameters for running connectivity models

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
        self.config = copy.deepcopy(config)
        self.config.update(**kwargs)

    def model_evaluate(self):
        """ Model evaluation routine on individual subject data, saved to disk       
        """

        # get model data: `X` and `Y` based on `train_inputs`
        model_data, model_fname, model_inputs=self._get_model_data()

        # get eval data: `X` and `Y` based on `eval_inputs`
        eval_data = self._get_eval_data()

        # get `X` and `Y` for evaluation data
        Y_eval = eval_data['eval_Y']
        X_eval = eval_data['eval_X']

        # split evaluation by `splitby` 
        # (conditions or tasks, depending on glm)
        splits = self._get_split_idx(X=X_eval)
        
        # initialise data dict
        data_dict_all = self._init_data_dict()

        # initialize weight dict for calculating noise ceilings
        weights_dict = AutoVivification()

        # loop over subjects
        for self.subj in self.config['eval_subjects']:

            print(f'evaluating model for s{self.subj:02} ...')

            # get betas for Y evaluation
            Y_eval_subj = Y_eval['betas'][f's{self.subj:02}']
            X_eval_subj = X_eval['betas'][f's{self.subj:02}']
                
            # get model weights
            model_weights = model_data[f's{self.subj:02}']['weights']

            if type(model_weights) != list:
                model_weights = [model_weights]

            # get model params (if they exist)
            self.param_name, self.param_values=self._get_model_params(model_inputs)

            # loop over model weights
            for self.i, model_weight in enumerate(model_weights):

                if self.config['eval_good_vox']:
                    # get "good voxels"
                    vox_idx = self._get_good_idx(Y=Y_eval_subj, W=model_weight)
                else:
                     # take all voxels
                    vox_idx = [True for i in np.arange(len(model_weight))]

                # loop over splits
                for self.split in splits:

                    Y_preds = {}
                    eval_idx = {}
                    for eval_mode in ['crossed', 'uncrossed']:

                        # get indices for X and Y for `eval_mode`: `crossed` or `uncrossed`
                        eval_idx[eval_mode] = self._get_eval_idx(eval_mode=eval_mode, X=X_eval)
                    
                        # calculate prediction between model weights and X evaluation data
                        X =  X_eval_subj[eval_idx[eval_mode]][splits[self.split]]

                        # calculate sparse matrix
                        weights = model_weight.T
                        if self.config['eval_sparse_matrix']:
                            weights = self._compute_sparse_matrix(W=model_weight.T)

                        # append weight dict    
                        param_value = self.param_values[self.i]
                        key = f'{self.param_name}_{param_value}'
                        weights_dict[key][f's{self.subj:02}']= weights

                        # scale eval data
                        if self.config['eval_scale']:
                            Y_preds[eval_mode] = self._predict(X=self._scale_data(X), W=weights)
                        else:
                            Y_preds[eval_mode] = self._predict(X=X, W=weights)

                    # get `crossed` and `uncrossed` eval and pred data
                    eval_Y_crossed = Y_eval_subj[eval_idx['crossed']][splits[self.split]][:, vox_idx]
                    eval_Y_uncrossed = Y_eval_subj[eval_idx['uncrossed']][splits[self.split]][:, vox_idx]
                    Y_pred_crossed = Y_preds['crossed'][:, vox_idx]
                    Y_pred_uncrossed = Y_preds['uncrossed'][:, vox_idx]

                    # scale the eval data
                    if self.config['eval_scale']:
                        eval_Y_crossed = self._scale_data(eval_Y_crossed)
                        eval_Y_uncrossed = self._scale_data(eval_Y_uncrossed)

                    # calculate sum-of-squares
                    ssq_all = self._calculate_ssq(eval_Y_crossed=eval_Y_crossed,
                                                eval_Y_uncrossed=eval_Y_uncrossed,
                                                Y_pred_crossed=Y_pred_crossed,
                                                Y_pred_uncrossed=Y_pred_uncrossed)

                    # calculate reliabilities
                    data_dict = self._calculate_reliabilities(ssq=ssq_all)

                    # calculate sparsity
                    data_dict.update(self._calculate_sparsity(W=weights))

                    # update data dict with/without voxel data
                    data_dict = self._update_data_dict(data_dict, Y_pred_uncrossed, Y_pred_crossed, weights)

                    # append data dict
                    data_dict_all = self._append_data_dict(data_dict, data_dict_all)

        if self.config['eval_noise_ceiling']:
            # calculate noise ceilings
            noise_ceiling_dict = self._calculate_noise_ceiling(data_dict=weights_dict)
            # update data dict
            data_dict_all.update(noise_ceiling_dict)

        # get eval params
        eval_params = copy.deepcopy(self.config)
        eval_params.update({'model_fname': model_fname, 'eval_splits': list(splits.keys())})

        # save eval parames to JSON and save eval predictions and reliabilities to HDF5
        self._save_eval_output(json_file=eval_params, hdf5_file=data_dict_all)
    
    def _get_model_data(self):
        """ Returns model training data based on requested model parameters from config file
        If a model has been run multiple times (i.e. different training parameters), this code
        searches through the timestamped JSON model files and returns the appropriate model
        in accordance with the requested parameters
        """
        dirs = Dirs(study_name=self.config['train_on'], glm=self.config['train_glm'])
        X_roi = self.config['train_X_roi']
        Y_roi = self.config['train_Y_roi']

        model_name = self.config['model_name']
        fname = f'{X_roi}_{Y_roi}_{model_name}_*.json'

        os.chdir(dirs.CONN_TRAIN_DIR)

        all_files = glob.glob(fname)
        diff_value = True
        diff_values = []
        for fname in all_files:
            fpath = os.path.join(dirs.CONN_TRAIN_DIR, fname)

            # load params from trained models
            trained_model = io.read_json(fpath)

            # get requested model parameters from config file
            requested_model = copy.deepcopy(self.config)

            # find common keys across trained model and requested model
            # and check for diff in keys
            common_keys = set(trained_model).intersection(set(requested_model))
            requested_model_common = {key: requested_model[key] for key in common_keys}
            trained_model_common = {key: trained_model[key] for key in common_keys}
            diff_value = list(diff(trained_model_common, requested_model_common))  
            diff_values.append(diff_value)    

            # if an appropriate model is found
            #  break out of loop           
            if not diff_value: 
                print(f'model was trained with the following parameters: \n {trained_model}')
                return io.read_hdf5(fpath.replace('.json', '.h5')), fname, trained_model

        raise Exception(f'There is no trained model with the parameters you requested \n train the model first before doing the evaluation \n for reference, here are the discrepencies between your requested parameters and existing model parameters: {diff_values}')
                                  
    def _get_model_params(self, model_inputs):
        model_params = [True for key in model_inputs if key=='model_params']
        if model_params:
            param_name = model_inputs['model_params']
            param_values = model_inputs[param_name]
        else:
            param_name = 'model_params'
            param_values = [None]
        return param_name, param_values
   
    def _get_eval_data(self):
        """ returns eval data based on `eval_inputs` set in __init__
        """
        dirs = Dirs(study_name=self.config['eval_on'], glm=self.config['eval_glm'])

        # get eval data: `X` and `Y`
        eval_data = {}
        if self.config['eval_stim'] == 'timeseries':
            self.data_type = {}
            self.data_type['roi'] = self.config[f'eval_X_roi']
            self.data_type['file_dir'] = self.config[f'eval_X_file_dir']
            self.experiment = [self.config['eval_on']]
            self.glm = self.config['eval_glm']
            self.stim = self.config['eval_stim']
            self.subjects = self.config['eval_subjects']
            self.sessions = self.config['eval_sessions']
            self.number_of_delays = self.config['train_number_of_delays']
            self.detrend = self.config['eval_detrend']
            self.structure = [self.config['eval_X_structure'], self.config['eval_Y_structure']]
            tempdata = self.get_conn_data()
            
                              
            eval_data[f'eval_X'] = tempdata['betas'][f'{self.config["train_X_structure"]}_undelayed'][f'{self.config["eval_on"]}']
            eval_data[f'eval_Y'] = tempdata['betas'][f'{self.config["eval_Y_structure"]}_delayed'][f'{self.config["eval_on"]}']
        else:
            for eval_input in ['X', 'Y']:

                # make sure you're setting correct eval params
                # don't love this code ...
                self.data_type = {}
                self.data_type['roi'] = self.config[f'eval_{eval_input}_roi']
                self.data_type['file_dir'] = self.config[f'eval_{eval_input}_file_dir']
                self.sessions = self.config['eval_sessions']
                self.experiment = [self.config['eval_on']]
                self.subjects = self.config['eval_subjects']
                self.incl_inst = self.config['eval_incl_inst']
                self.glm = self.config['eval_glm']
                self.stim = self.config['eval_stim']
                self.scale = self.config['eval_scale']
                self.avg = self.config['eval_avg']
                self.subtract_sess_mean = self.config['eval_subtract_sess_mean']
                self.subtract_exp_mean = self.config['eval_subtract_exp_mean']

                eval_data[f'eval_{eval_input}'] = self.get_conn_data()

        return eval_data
    
    def _predict(self, X, W): 
        # get `Y_pred`(X*B) based on model weights
        return np.matmul(X, W)
                
    def _get_good_idx(self, Y, W):
        # CHECK
        # is there a better way to write this function? should it be np.nansum??
        return (np.sum(abs(Y), axis = 0) > 0) * ((np.isnan(np.sum(Y, axis = 0)) * 1) == 0) * ((np.isnan(np.sum(W, axis = 1)))*1 == 0)

    def _get_eval_idx(self, eval_mode, X):
        # get indices for eval mode: `crossed` or `uncrossed`
        
        eval_stim = self.config['eval_stim']
        
        if eval_stim =='timeseries':
            if eval_mode =='crossed':
                indices = str(2)
            else:
                indices = str(1)
            return indices
        else:
            stims = X[f'{eval_stim}Num']
            sessions = X['sess'].astype(int) 

            indices = []
            for stim, sess in zip(stims, sessions):

                if sess == 1:
                    indices.append(stim)
                elif sess == 2:
                    indices.append(stim + max(stims) + 1)

            # get indices if eval_mode is `crossed`
            if eval_mode == 'crossed':
                indices = [*indices[-sessions.tolist().count(2):], *indices[:sessions.tolist().count(1)]] 

            return np.array(indices)
    
    def _get_split_idx(self, X):
        split_dict = {}

        if self.config['eval_splitby']:
            eval_splitby = self.config['eval_splitby']
            splits = X[f'{eval_splitby}Num']
            split_names = X[f'{eval_splitby}Name']
            for (num, name) in zip(splits, split_names):
                # get split index
                split_dict[name] = [True if x == num else False for x in splits]
        else: 
            eval_stim = self.config['eval_stim']
            splits = X[f'{eval_stim}Num']
            split_dict = {f'all-{eval_stim}': [True for x in splits]}
    
        return split_dict

    def _compute_sparse_matrix(self, W):
        n_max = 1
        sparse_mat = W * (W >= np.sort(abs(W), axis=0)[[-n_max],:]).astype(int)

        return sparse_mat
    
    def _calculate_ssq(self, eval_Y_crossed, eval_Y_uncrossed, Y_pred_crossed, Y_pred_uncrossed):
        # evaluation output
        ssq_pred = np.nansum(Y_pred_uncrossed**2, axis=0) # sum-of-squares of predictions
        ssq_y = np.nansum(eval_Y_crossed**2, axis=0) # sum-of-squares of Y evaluation data
        ssc_pred = np.nansum(Y_pred_uncrossed * Y_pred_crossed, axis=0) # covariance of predictions
        ssc_y = np.nansum(eval_Y_uncrossed * eval_Y_crossed, axis=0) # covariance of Y evaluation data
        ssc_ncv = np.nansum(Y_pred_crossed * eval_Y_crossed, axis=0) # covariance of uncrossed prediction and Y evaluation data
        ssc_cv = np.nansum(Y_pred_uncrossed * eval_Y_crossed, axis=0) # covariance of crossed prediction and Y evaluation data
        
        return {'ssq_pred': ssq_pred, 'ssq_y': ssq_y, 'ssc_pred': ssc_pred,
                'ssc_y': ssc_y, 'ssc_ncv': ssc_ncv, 'ssc_cv': ssc_cv
                }
    
    def _calculate_reliabilities(self, ssq): 
        # calculate reliabilities
        R_pred_cv_vox = ssq['ssc_cv'] / np.sqrt(ssq['ssq_y'] * ssq['ssq_pred']) # crossed predictive correlation
        R_pred_ncv_vox = ssq['ssc_ncv'] / np.sqrt(ssq['ssq_y'] * ssq['ssq_pred']) # uncrossed predictive correlation
        R_y_vox = ssq['ssc_y'] / ssq['ssq_y'] # reliability of Y evaluation data
        R_pred_vox = ssq['ssc_pred'] / ssq['ssq_pred'] # reliability of prediction

        data_dict = {}
        if self.config['eval_save_maps']:
            data_dict = {'R_pred_cv_vox': R_pred_cv_vox, 'R_pred_ncv_vox': R_pred_ncv_vox,
                        'R_y_vox': R_y_vox, 'R_pred_vox': R_pred_vox}

        data_dict.update({'R_pred_cv': np.nanmean(R_pred_cv_vox, axis=0),
                        'R_pred_ncv': np.nanmean(R_pred_ncv_vox, axis=0),
                        'R_y': np.nanmean(R_y_vox, axis=0),
                        'R_pred': np.nanmean(R_pred_vox, axis=0)})

        return data_dict
    
    def _calculate_sparsity(self, W):
        sparsity_dict = {}

        # calculate total % of sum weights
        W_sort = np.sort(abs(W), axis=0)
        W_sort_standarized = np.divide(W_sort, sum(W_sort))   
        sparsity_vox = np.nanmax(W_sort_standarized, axis=0)
        wta_vox = np.nanargmax(abs(W), axis=0) + 1

        # calculate ginni
        num_feat, _ = W_sort_standarized.shape
        weight = (num_feat - (np.arange(0,num_feat)).T + 0.5) / num_feat 
        ginni_vox = 1 - 2*(np.matmul(W_sort_standarized.T, weight) ); 

        # save to dict
        sparsity_dict = {'S_best_weight': np.nanmean(sparsity_vox), 'S_ginni': np.nanmean(ginni_vox)}
        if self.config['eval_save_maps']:
            sparsity_dict.update({'S_best_weight_vox': sparsity_vox, 'S_ginni_vox': ginni_vox, 'wta_vox': wta_vox})

        return sparsity_dict
    
    def _calculate_noise_ceiling(self, data_dict):
        
        # initialize weight dict
        weight_dict_all = self._init_data_dict()

        # loop over model param
        for key in data_dict:

            def _append_weights(data_dict, key):
                # loop over subj
                weights_all = []
                for subj in data_dict[key]:
                    weights_all.append(data_dict[key][subj])
                return np.array(weights_all)

            weights_all = _append_weights(data_dict, key)

            # calculate noise ceilings for each subj
            subj_list = np.arange(0,len(weights_all))
            for subj in subj_list:
                
                # get idx of all other subjects
                other_subjs = subj_list[subj_list!=subj]

                # get average of all other subjects
                avg_other_subjs = np.nanmean(weights_all[other_subjs], axis=0)

                # calculate high noise ceiling
                R_high, R_high_vox = self._calculate_R(X=weights_all[subj], Y=avg_other_subjs)

                # calculate low noise ceiling
                # low_noise_dict = defaultdict(partial(np.ndarray, 0))
                R_noise_low = []
                R_noise_low_vox = []
                for other_subj in other_subjs:
                    R_low, R_low_vox = self._calculate_R(X=weights_all[subj], Y=weights_all[other_subj])
                    R_noise_low.append(R_low)
                    R_noise_low_vox.append(R_low_vox)

                # append noise ceilings to dict
                weight_dict = {'R_noise_high': R_high, 
                               'R_noise_high_vox': R_high_vox, 
                               'R_noise_low': np.nanmean(R_noise_low, axis=0), 
                               'R_noise_low_vox': np.nanmean(R_noise_low_vox, axis=0)}

                weight_dict_all = self._append_data_dict(weight_dict, weight_dict_all)

        return weight_dict_all

    def _calculate_R(self, X, Y):
        # Calculating R between `X` and `Y`

        SYP = np.nansum(X*Y, axis=0)
        SPP = np.nansum(Y*Y, axis=0)
        # SST = np.sum((X - X.mean()) ** 2, axis=0) # use np.nanmean(Y) here?
        tmp = X - np.nanmean(X, axis=0)
        SST = np.nansum(tmp**2, axis=0)

        R = np.nansum(SYP)/np.sqrt(np.nansum(SST)*np.nansum(SPP))
        R_vox = SYP/np.sqrt(SST*SPP) # per voxel

        return R, R_vox

    def _calculate_R2(self, X, Y):
        # Calculating R2
        res = X - Y

        SSR = np.nansum(res **2, axis = 0) # remember: without setting the axis, it just "flats" out the whole array and sum over all
        # SST = np.sum((X - X.mean()) ** 2, axis = 0) # use np.nanmean(Y) here??
        tmp = X - np.nanmean(X, axis=0)
        SST = np.nansum(tmp**2, axis=0)

        R2 = 1 - (np.nansum(SSR)/np.nansum(SST))
        R2_vox = 1 - (SSR/SST)

        return R2, R2_vox
    
    def _update_data_dict(self, data_dict, Y_pred_uncrossed, Y_pred_crossed, weights):
        # add predictions to data dict
        if self.config['eval_save_maps']:
            data_dict.update({'Y_pred_ncv_vox': np.nanmean(Y_pred_uncrossed, axis=0),
                            'Y_pred_cv_vox': np.nanmean(Y_pred_crossed, axis=0),
                            'weights': weights})
        else:
            # add subjects, splits, lambdas
            data_dict.update({'eval_splits': self.split, self.param_name: self.param_values[self.i], 'eval_subjects': self.subj})

        return data_dict
    
    def _init_data_dict(self):
        data_dict_all = defaultdict(partial(np.ndarray, 0))
        
        if self.config['eval_save_maps']:
            data_dict_all = AutoVivification()

        return data_dict_all
    
    def _append_data_dict(self, data_dict, data_dict_all):
        # append data dict with/without voxel data
        if not self.config['eval_save_maps']:
            data_dict = {k:v for k,v in data_dict.items() if 'vox' not in k} 
            for k,v in data_dict.items():
                # data_dict[k].append(v)
                data_dict_all[k] = np.append(data_dict_all[k], v)
        else:
            param_value = self.param_values[self.i]
            model_key = f'{self.param_name}_{param_value}'
            data_dict = {k:v for k,v in data_dict.items() if 'vox' in k} 
            data_dict_all[f's{self.subj:02}'][model_key][self.split] = data_dict

        return data_dict_all
    
    def _get_outpath(self, file_type, **kwargs):
        """ sets outpath for connectivity evaluation output
            Args: 
                file_type (str): 'json' or 'h5' 
            Returns: 
                full path to connectivity output for model evaluation
        """
        dirs = Dirs(study_name=self.config['eval_on'], glm=self.config['eval_glm'])

        # define eval name
        X_roi = self.config['eval_X_roi']
        Y_roi = self.config['eval_Y_roi']

        if kwargs.get('timestamp'):
            timestamp = kwargs['timestamp']
        else:
            timestamp = f'{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}'

        model_name = self.config['model_name']
        fname = f'{X_roi}_{Y_roi}_{model_name}_{timestamp}{file_type}'
        fpath = os.path.join(dirs.CONN_EVAL_DIR, fname)
        
        return fpath, timestamp
    
    def _save_eval_output(self, json_file, hdf5_file):
        out_path, timestamp = self._get_outpath(file_type='.json')
        io.save_dict_as_JSON(fpath=out_path, data_dict=json_file)

        # save model data to HDF5
        out_path, _ = self._get_outpath(file_type='.h5', timestamp=timestamp)
        io.save_dict_as_hdf5(fpath=out_path, data_dict=hdf5_file)

    
