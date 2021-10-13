import click
import os
import pandas as pd
import nibabel as nib

from connectivity import weight_maps as cmaps
from connectivity import visualize as summary
import connectivity.constants as const

# @click.command()
# @click.option("--roi")
# @click.option("--weights")
# @click.option("--data_type")

def distances_summary(
    atlas='MDTB10', 
    weights='absolute', 
    method='ridge', # L2regression
    thresholds=[1,3,5],
    metric='gmean'
    ):

    exp = 'sc1'
    dirs = const.Dirs(exp_name=exp)

    subjs, _ = cmaps.split_subjects(const.return_subjs, test_size=0.3)

    # models, cortex_names = summary.get_best_models(method=method) 
    cortex = 'tessels1002'
    models = [f'{method}_{cortex}_alpha_8']
    cortex_names = ['tessels1002']

    for (best_model, cortex) in zip(models, cortex_names):
        
        # full path to best model
        fpath = os.path.join(dirs.conn_train_dir, best_model)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        for threshold in thresholds:
            dataframe_all = pd.DataFrame()
            for subj in subjs:
                roi_betas, reg_names, colors = cmaps.average_region_data(subj,
                                        exp=exp, cortex=cortex, 
                                        atlas=atlas, method=method, alpha=alpha, 
                                        weights=weights, average_subjs=False)

                # save out cortical distances
                dataframe = cmaps.distances_cortex(roi_betas, reg_names, colors, 
                            cortex=cortex, threshold=threshold, metric=metric)
                dataframe['subj'] = subj
                dataframe['atlas'] = atlas
                dataframe['metric'] = metric
                dataframe_all = pd.concat([dataframe_all, dataframe])
            dataframe_all.to_csv(os.path.join(fpath, f'distances_summary_{cortex}_{atlas}_threshold_{threshold}_{metric}.csv'))

def distances_map(
    atlas='MDTB10', 
    method='ridge', # L2regression
    thresholds=[1,2,3,4,5]
    ):

    exp = 'sc1'
    dirs = const.Dirs(exp_name=exp)

    subjs, _ = cmaps.split_subjects(const.return_subjs, test_size=0.3)

    # models, cortex_names = summary.get_best_models(method=method) 
    cortex = 'tessels1002'
    models = [f'{method}_{cortex}_alpha_8']
    cortex_names = ['tessels1002']

    for (best_model, cortex) in zip(models, cortex_names):
        
        # full path to best model
        fpath = os.path.join(dirs.conn_train_dir, best_model)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        # get alpha for each model
        alpha = int(best_model.split('_')[-1])
        roi_betas, reg_names, _ = cmaps.average_region_data(subjs,
                                exp=exp, cortex=cortex, 
                                atlas=atlas, method=method, alpha=alpha, 
                                weights='nonzero', average_subjs=True)

        # save out cortical maps
        for threshold in thresholds:
            giis = cmaps.regions_cortex(roi_betas, reg_names, cortex=cortex, threshold=threshold)
            fname = f'group_{method}_{cortex}_{atlas}_threshold_{threshold}'
            [nib.save(gii, os.path.join(fpath, f'{fname}.{hem}.func.gii')) for (gii, hem) in zip(giis, ['L', 'R'])]

def run():
    distances_summary()
    distances_map()

# if __name__ == "__main__":
#     run()