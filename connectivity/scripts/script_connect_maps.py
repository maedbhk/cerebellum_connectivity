import click
import os
import SUITPy.flatmap as flatmap
import nibabel as nib

from connectivity import connect_maps as cmaps
from connectivity import visualize as summary
import connectivity.constants as const

@click.command()
@click.option("--atlas")

def run(atlas='MDTB_10Regions'):

    cerebellum_fpath = os.path.join(flatmap._base_dir, 'example_data', f'{atlas}.nii')

    for exp in range(2):

        dirs = const.Dirs(exp_name=f"sc{2-exp}")
    
        # get best model (for each method and parcellation)
        models, cortex_names = summary.get_best_models(train_exp=f"sc{2-exp}")

        for (best_model, cortex) in zip(models, cortex_names):
            
            # full path to best model
            fpath = os.path.join(dirs.conn_train_dir, best_model)

            # save voxel/vertex maps for best training weights (for group parcellations only)
            # if 'wb_indv' not in cortex:
            #     cmaps.weight_maps(model_name=best_model, cortex=cortex, train_exp=f"sc{2-exp}")

            if 'lasso' in best_model:
                
                cmaps.lasso_maps_cerebellum(model_name=best_model, 
                                            train_exp=f"sc{2-exp}") 

                func_giis, hem_names = cmaps.lasso_maps_cortex(model_name=best_model, 
                                        train_exp=f"sc{2-exp}", 
                                        cortex=cortex, 
                                        cerebellum_fpath=cerebellum_fpath
                                        ) 

                for (func_gii, hem) in zip(func_giis, hem_names):
                    nib.save(func_gii, os.path.join(fpath, f'group_lasso_cortex_{atlas}.{hem}.func.gii'))

if __name__ == "__main__":
    run()