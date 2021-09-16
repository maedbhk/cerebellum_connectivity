import click
import os
import SUITPy.flatmap as flatmap
import nibabel as nib

from connectivity import weight_maps as cmaps
from connectivity import visualize as summary
import connectivity.constants as const

@click.command()
@click.option("--atlas")
@click.option("--weights")
@click.option("--data_type")

def lasso_maps(
    atlas='MDTB_10Regions', 
    weights='positive', 
    data_type='label'
    ):
    """ creates cortical connectivity maps for lasso (functional and lasso)

    Args: 
        atlas (str): default is 'MDTB_10Regions'
        weights (str): 'positive' or 'absolute'. default is 'positive'
        data_type (str): 'func' or 'label'. default is 'label'
    """

    cerebellum_nifti = os.path.join(flatmap._base_dir, 'example_data', f'{atlas}.nii')
    cerebellum_gifti = os.path.join(flatmap._base_dir, 'example_data', f'{atlas}.label.gii')

    # for exp in range(2):
    for exp in range(2):

        dirs = const.Dirs(exp_name=f"sc{2-exp}")
    
        # get best model (for each method and parcellation)
        models, cortex_names = summary.get_best_models(train_exp=f"sc{2-exp}")

        for (best_model, cortex) in zip(models, cortex_names):
            
            # full path to best model
            fpath = os.path.join(dirs.conn_train_dir, best_model)

            if 'lasso' in best_model:
                
                cmaps.lasso_maps_cerebellum(model_name=best_model, 
                                            train_exp=f"sc{2-exp}",
                                            weights=weights) 

                giis, hem_names = cmaps.lasso_maps_cortex(model_name=best_model, 
                                        train_exp=f"sc{2-exp}", 
                                        cortex=cortex, 
                                        cerebellum_nifti=cerebellum_nifti,
                                        cerebellum_gifti=cerebellum_gifti,
                                        weights=weights,
                                        data_type=data_type
                                        ) 
                # probabilistic data are functional data
                fname = f'group_lasso_{weights}_{atlas}_cortex'
                if data_type=='prob':
                    data_type='func'
                    fname = f'group_lasso_{weights}_{atlas}_prob_cortex'

                for (gii, hem) in zip(giis, hem_names):
                    nib.save(gii, os.path.join(fpath, f'{fname}.{hem}.{data_type}.gii'))

                cmaps.weight_maps(model_name=best_model, cortex=cortex, train_exp=f"sc{2-exp}")

def run(
    atlas='MDTB_10Regions', 
    weights='positive', 
    data_type='label'
    ):
    
    # # generate lasso maps
    # lasso_maps(atlas,  weights, data_type)

    # # generate weight maps for cortex and cerebellum
    # for exp in range(2):
    #     # get best model (for each method and parcellation)
    #     models, cortex_names = summary.get_best_models(train_exp=f"sc{2-exp}")
    #     for (best_model, cortex) in zip(models, cortex_names):
    #         # save voxel/vertex maps for best training weights (for group parcellations only)
    #         if 'wb_indv' not in cortex:
    #             cmaps.weight_maps(model_name=best_model, cortex=cortex, train_exp=f"sc{2-exp}")

    # save out np array of best weights
    for exp in ['sc1', 'sc2']:
        cmaps.best_weights(train_exp=exp, method='L2regression')

if __name__ == "__main__":
    run()