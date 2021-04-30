# import libraries
import click

from connectivity.scripts import script_mk

@click.command()
@click.option("--train_exp")

def run(train_exp='sc1'):

    rois = ['tessels1002']
    params = [1,2,3,4,5,10]

    for roi in rois:

        for param in params:
            
            script_mk.save_weight_maps(model_name=f'NTakeAll_{roi}_{param}_positive', cortex=roi, train_exp=train_exp)

if __name__ == "__main__":
    run()