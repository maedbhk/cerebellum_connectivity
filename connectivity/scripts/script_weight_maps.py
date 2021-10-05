import click
import os
from SUITPy import atlas, flatmap
import nibabel as nib

from connectivity import weight_maps as cmaps
from connectivity import visualize as summary
import connectivity.constants as const

def lasso_maps(
    atlas='MDTB10', 
    weights='positive', 
    data_type='func',
    exp='sc1',
    ):
    """ creates cortical connectivity maps for lasso (functional and lasso)

    Args: 
        atlas (str): default is 'MDTB10'
        weights (str): 'positive' or 'absolute'. default is 'positive'
        data_type (str): 'func' or 'label'. default is 'func'
    """
    dirs = const.Dirs(exp_name=exp)

    dirs = const.Dirs()
    atlas.fetch_king_2019(data='atl', data_dir=dirs.cerebellar_atlas_dir())

    cerebellum_nifti = os.path.join(dirs.cerebellar_atlas_dir, 'king_2019', f'{atlas}.nii')
    cerebellum_gifti = os.path.join(dirs.cerebellar_atlas_dir, 'king_2019', f'{atlas}.label.gii')

    models = ['lasso_tessels1002_alpha_-2']
    cortex_names = ['tessels1002']

    for (best_model, cortex) in zip(models, cortex_names):
        
        # full path to best model
        fpath = os.path.join(dirs.conn_train_dir, best_model)

        if 'lasso' in best_model:
            
            # cmaps.lasso_maps_cerebellum(model_name=best_model, 
            #                             train_exp=exp,
            #                             weights=weights) 

            # get alpha for each model
            alpha = int(best_model.split('_')[-1])
            giis, hem_names = cmaps.lasso_maps_cortex( 
                                    train_exp=exp, 
                                    cortex=cortex, 
                                    alpha=alpha,
                                    atlas=atlas,
                                    weights=weights,
                                    data_type=data_type
                                    ) 
            # fname
            fname = f'group_lasso_{weights}_{atlas}_cortex'

            for (gii, hem) in zip(giis, hem_names):
                nib.save(gii, os.path.join(fpath, f'{fname}.{hem}.{data_type}.gii'))

            cmaps.weight_maps(model_name=best_model, cortex=cortex, train_exp=exp)

# @click.command()
# @click.option("--atlas")
# @click.option("--weights")
# @click.option("--data_type")

def run(
    atlas='MDTB10', 
    weights='positive', 
    data_type='func'
    ):
    
    # generate lasso maps
    lasso_maps(atlas,  weights, data_type, exp='sc1')

    # # generate weight maps for cortex and cerebellum
    # weight_maps()

    # # save out np array of best weights
    # for exp in ['sc1', 'sc2']:
    #     cmaps.best_weights(train_exp=exp, method='L2regression', save=True)

# if __name__ == "__main__":
#     run()