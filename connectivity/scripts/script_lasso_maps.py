import click

from connectivity import weight_maps as cmaps
from connectivity import visualize as summary

@click.command()
@click.option("--roi")
@click.option("--weights")
@click.option("--data_type")

def run(
    exp='sc1',
    weights='absolute', 
    method='ridge', # L2regression
    ):

    models, cortex_names = summary.get_best_models(method=method) 

    for best_model in models:

        cmaps.lasso_maps_cerebellum(model_name=best_model, 
                                    train_exp=exp,
                                    weights=weights)

        # save out np array of best weights
        for exp in ['sc1', 'sc2']:
            cmaps.best_weights(train_exp=exp, method=method, save=True)

if __name__ == "__main__":
    run()