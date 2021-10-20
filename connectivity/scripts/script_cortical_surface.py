import click
import pandas as pd

from connectivity import weights as cweights
from connectivity import visualize as summary
import connectivity.constants as const

@click.command()
@click.option("--exp")
@click.option("--weights")
@click.option("--method")

def run(
    exp='sc1',
    weights='nonzero', 
    method='lasso', # L2regression
    ):

    dirs = const.Dirs(exp_name=exp)
    models, _ = summary.get_best_models(method=method) 

    df_all = pd.DataFrame()
    for best_model in models:

        df = cweights.cortical_surface_voxels(model_name=best_model, 
                                    train_exp=exp,
                                    weights=weights,
                                    save_maps=False)
        df_all = pd.concat([df_all, df])
        
        # cweights.cortical_surface_rois(model_name=best_model, 
        #                             train_exp=exp,
        #                             weights=weights)

    # save to disk
    df_all.to_csv(dirs.conn_train_dir, f'cortical_surface_stats.csv')

if __name__ == "__main__":
    run()