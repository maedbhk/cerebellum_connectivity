import click
import pandas as pd
import os
from matplotlib import cm
import connectivity.constants as const
import connectivity.nib_utils as nio
from connectivity import data as cdata
from connectivity import io 

def run(glm='glm7', metric='R', methods=['WTA', 'ridge']):
    """
    Args: 
        glm (str): 'glm7'
        metric (str): evaluation metric: 'R' or 'R2'. default is 'R'
        methods (list of str): default is ['WTA', 'ridge']

    Returns: 
        saves nifti and gifti for difference map between `methods` for evaluated models
    """
    # loop over experiments
    for exp in range(2):
        
        # get eval summary
        dirs = const.Dirs(exp_name=f"sc{exp+1}", glm=glm)
        df = pd.read_csv(os.path.join(dirs.conn_eval_dir, 'eval_summary.csv'))
        df = df[['name', 'X_data']].drop_duplicates() # get unique model names

        # get outpath
        fpath = os.path.join(dirs.conn_eval_dir, 'model_comparison')
        io.make_dirs(fpath)

        # loop over cortical parcellations
        for cortex in df['X_data'].unique():

            # grab full paths to trained models for `cortex` and filter out `methods`
            imgs = [os.path.join(dirs.conn_eval_dir, model, f'group_{metric}_vox.nii') for model in df['name'] if cortex in model] 
            imgs = [img for img in imgs if any(k in img for k in methods)]

            # get binarized difference map
            nib_obj = nio.binarize_vol(imgs, mask=cdata.read_mask(), metric='max')

            # save to disk
            colormap = cm.get_cmap('tab10', len(methods)+1)
            label_names = ['label-01']
            label_names.extend(methods)
            colormap.colors[0] = [0,0,0,1] # assign zero label
            nio.make_gifti_cerebellum(data=nib_obj[0], 
                                    mask=cdata.read_mask(),
                                    outpath=os.path.join(fpath, f'group_difference_{metric}_{cortex}.label.gii'),
                                    stats=None,
                                    data_type='label',
                                    label_names=label_names,
                                    label_RGBA=colormap.colors
                                    )
                        
if __name__ == "__main__":
    run()
