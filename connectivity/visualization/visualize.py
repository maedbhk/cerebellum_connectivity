import os 
import pandas as pd
import numpy as np
import re
import glob

import seaborn as sns
import matplotlib.pyplot as plt
from collections import MutableMapping
from collections import defaultdict

# import plotly.graph_objects as go

from connectivity.constants import Dirs, Defaults
from connectivity import io
from connectivity.data.prep_data import DataManager

"""
Created on Wed 26 13:31:34 2020
Visualization routine for connectivity models

@author: Maedbh King and Ladan Shahshahani
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
    
    def get_all_files(self, file):
        return glob.glob(os.path.join(self.dirs.CONN_EVAL_DIR, f'*{file}*.json'))
    
    def read_to_dataframe(self, files):

        # get data for repeat models
        df_all = pd.DataFrame()
        for file in files:
            
            # load data file
            data_dict = self._load_data_file(data_fname=file.replace('json', 'h5'))

            # load param file
            param_dict = self._load_param_file(param_fname=file)

            # flatten nested json dict
            param_dict = self._flatten_nested_dict(data_dict=param_dict)

            # convert json and hdf5 to dataframes
            df_param = self._convert_to_dataframe(data_dict=param_dict)
            df_data = pd.DataFrame.from_dict(data_dict)

            # merge param and data 
            df_merged = df_param.merge(df_data)

            # concat repated models
            df_all = pd.concat([df_all, df_merged], axis=0)

        # tidy up dataframe
        cols_to_stack = [col for col in df_all.columns if 'R_' in col]
        df1 = pd.concat([df_all]*len(cols_to_stack)).reset_index(drop=True)
        df2 = pd.melt(df_all[cols_to_stack]).rename({'variable': 'R_type', 'value': 'R'}, axis=1)
        df_all = pd.concat([df1, df2], axis=1)

        return df_all

class Predictions(Utils):

    def __init__(self, model_name = 'tesselsWB162_grey_nan_l2_regress', eval_on = 'sc2', glm = 7):
        self.model_name = model_name
        self.eval_on = eval_on
        self.glm = glm

        self.dirs = Dirs(study_name = self.eval_on, glm = self.glm)

    def load_dataframe(self):
        fnames = self.get_all_files(file=self.model_name)

        dataframe = self.read_to_dataframe(files=fnames)

        return dataframe

    def plot_prediction_group(self, dataframe):
        
        sns.set(rc={'figure.figsize':(20,10)})
        sns.factorplot(x='lambdas', y='R', hue='R_type', data=dataframe)
        plt.xlabel('lambdas', fontsize=20),
        plt.ylabel('R', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        # plt.ylim(bottom=.7, top=1.0)

        plt.show()

    def plot_prediction_tasks(self, dataframe):
        sns.set(rc={'figure.figsize':(20,10)})
        sns.factorplot(x='lambdas', y='R_pred', hue='eval_splits', data=dataframe)
        plt.xlabel('lambdas', fontsize=20),
        plt.ylabel('R', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        # plt.ylim(bottom=.7, top=1.0)

        plt.show()

class Betas(DataManager):
    
    def __init__(self):
        super().__init__()
        self.inputs = {'X': {'roi': 'tesselsWB162', 'file_dir': 'encoding', 'structure': 'cortex'},
                       'Y': {'roi': 'grey_nan', 'file_dir': 'encoding', 'structure': 'cerebellum'}}
        self.scale = True
        self.defaults = Defaults()
        self.subjects = self.defaults.return_subjs

    def load_dataframe(self):
        data ={}
        for roi in self.inputs:
            
            self.data_type = self.inputs[roi]
            data[roi] = self.get_conn_data()

        # get avg betas for cortex and cerebellum
        roi_dict = self._get_avg_betas(data_dict=data)

        # horizontally concat dataframes
        X = pd.DataFrame.from_dict(roi_dict['X'])
        Y = pd.DataFrame.from_dict(roi_dict['Y'])
        dataframe = pd.concat([X, Y], axis=1)

        # drop duplicates
        dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()] # drop duplicate cols

        # save avg betas to disk
        dirs = Dirs()
        dataframe.to_csv(os.path.join(dirs.BASE_DIR, 'task_betas.csv'), index=False)

        return dataframe

    def _get_avg_betas(self, data_dict):

        # loop over rois
        roi_dict = {}
        for roi in data_dict:

            structure = self.inputs[roi]['structure']
            roi_name = self.inputs[roi]['roi']
            
            avg_betas = defaultdict(list)
            for subj in self.subjects:

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

    def task_scatter(self, dataframe):

        dataframe = dataframe[['stim_name', 'cortex_beta', 'cerebellum_beta']].groupby('stim_name').agg('mean').reset_index()

        sns.set(rc={'figure.figsize':(10,10)})
        sns.scatterplot(x="cortex_beta", y="cerebellum_beta", hue=dataframe['stim_name'].tolist(), data=dataframe)

        # plot regression line
        m, b = np.polyfit(dataframe['cortex_beta'], dataframe['cerebellum_beta'], 1)
        plt.plot(dataframe['cortex_beta'], m*dataframe['cortex_beta']+ b)

        plt.xlabel('cortex', fontsize=20),
        plt.ylabel('cerebellum', fontsize=20)
        plt.title('', fontsize=20);
        plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
        # plt.ylim(bottom=.7, top=1.0)

        plt.show()

    def task_scatter_interactive(self, dataframe):
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
            title_text='Avg. betas: Cortex and Cerebellum'
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