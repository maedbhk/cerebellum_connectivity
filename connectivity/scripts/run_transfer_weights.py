import os
import pandas as pd
import glob
import click
from pathlib import Path
from shutil import copyfile
import deepdish as dd
import numpy as np

def _remap():

    return {'Schaefer_7_100': 'Schaefer_2018_7Networks_100',
            'Schaefer_7_200': 'Schaefer_2018_7Networks_200',
            'Schaefer_7_300': 'Schaefer_2018_7Networks_300', 
            'arslan_100': 'Arslan_1_100', 
            'arslan_200': 'Arslan_1_200',
            'arslan_250': 'Arslan_1_250',
            'arslan_50': 'Arslan_1_50', 
            'fan': 'Fan_2016', 
            'gordon': 'Gordon',
            'mdtb1002_007': 'mdtb1002_007',
            'mdtb1002_025': 'mdtb1002_025',
            'mdtb1002_050': 'mdtb1002_050',
            'mdtb1002_100': 'mdtb1002_100',
            'mdtb1002_150': 'mdtb1002_150',
            'mdtb1002_300': 'mdtb1002_300',
            'mdtb1002_400': 'mdtb1002_400',
            'mdtb1002_500': 'mdtb1002_500',
            'shen': 'Shen',
            'tessels0042': 'Icosahedron-42',
            'tessels0162': 'Icosahedron-162',
            'tessels0362': 'Icosahedron-362',
            'tessels0642': 'Icosahedron-642',
            'tessels1002': 'Icosahedron-1002',
            'yeo17': 'Yeo_JNeurophysiol11_17Networks',
            'yeo7': 'Yeo_JNeurophysiol11_7Networks'
            }

@click.command()
@click.option("--connect_dir")
@click.option("--learn_dir")

def run(connect_dir, learn_dir):

    mdtb_dir = os.path.join(learn_dir, 'mdtb')

    print('working')

    if not os.path.exists(mdtb_dir):
        os.makedirs(mdtb_dir)

    if not os.path.exists(connect_dir):
        os.makedirs(connect_dir)

    # navigate to connectivity weight dir and grab files
    os.chdir(connect_dir)
    files = glob.glob('*.h5')
    print(files)

    # get remapping of file names
    data_dict = _remap()

    # transfer best models
    best_models = [f for f in files if 'best_model' in f]

    # load names of models and filter based on `_remap()`
    df_all = pd.DataFrame()
    for model in best_models:
        df = pd.read_csv(os.path.join(connect_dir, model)) 
        df_all = pd.concat([df, df_all])
    df_filter = df_all[df_all['cortex_names'].isin(data_dict.keys())]
    df_filter.to_csv(os.path.join(mdtb_dir, 'best_models.csv'))

    # copy best weights from connectivity to learning dir (change filenames)
    for file in files:
        for k,v in data_dict.items():
            if k in file:
                if 'ridge' in file:
                    fname = file.replace(k, v).replace('ridge', 'RIDGE')
                elif 'lasso' in file:
                    fname = file.replace(k, v).replace('lasso', 'LASSO')
                else:
                    fname = file.replace(k, v)

                src = os.path.join(connect_dir, file)
                dest = os.path.join(mdtb_dir, Path(fname).stem + '_mdtb.h5')
                copyfile(src, dest)

                # transpose the data first
                data = dd.io.load(dest)
                data['weights'] = data['weights'].T
                dd.io.save(dest, data)

                print('worked')

if __name__ == "__main__":
    run()

