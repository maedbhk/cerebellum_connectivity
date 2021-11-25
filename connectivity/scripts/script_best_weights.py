import click

from connectivity import weights as cmaps

@click.command()
@click.option("--exp")
@click.option("--method")

def run(
    exp='sc1',
    method='ridge',
    ):

    # save out np array of best weights
    cmaps.best_weights(train_exp=exp, method=method, save=True)

if __name__ == "__main__":
    run()