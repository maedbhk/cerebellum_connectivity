import click
import os
import SUITPy.flatmap as flatmap

from connectivity import connect_maps as cmaps
from connectivity import visualize as summary

@click.command()
@click.option("--atlas")

def run(atlas='MDTB_10Regions'):

    cerebellum_fpath = os.path.join(flatmap._base_dir, 'example_data', f'{atlas}.nii')

    for exp in range(2):
    
        # get best model (for each method and parcellation)
        models, cortex_names = summary.get_best_models(train_exp=f"sc{2-exp}")

        for (best_model, cortex) in zip(models, cortex_names):
            
            # save voxel/vertex maps for best training weights (for group parcellations only)
            # if 'wb_indv' not in cortex:
            #     cmaps.weight_maps(model_name=best_model, train_exp=f"sc{2-exp}")

            if 'lasso' in best_model:
                cmaps.lasso_maps_cerebellum(model_name=best_model, 
                                            train_exp=f"sc{2-exp}") 

                cmaps.lasso_maps_cortex(model_name=best_model, 
                                    train_exp=f"sc{2-exp}", 
                                    cortex=cortex, 
                                    cerebellum_fpath=cerebellum_fpath) 

if __name__ == "__main__":
    run()