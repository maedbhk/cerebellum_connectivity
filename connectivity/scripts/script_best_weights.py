import click

from connectivity import weights as cmaps

@click.command()
@click.option("--roi")
@click.option("--weights")
@click.option("--data_type")

def run(
    method='ridge', # L2regression
    ):

    # save out np array of best weights
    for exp in ['sc1', 'sc2']:
        cmaps.best_weights(train_exp=exp, method=method, save=True)

if __name__ == "__main__":
    run()