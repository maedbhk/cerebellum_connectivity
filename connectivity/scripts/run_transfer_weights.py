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
    # connect_dir = '/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/best_weights'
    # learn_dir = '/global/scratch/users/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/derivatives/conn_models/train'

    mdtb_dir = os.path.join(learn_dir, 'mdtb')
    mdtb_smooth_dir = os.path.join(learn_dir, 'mdtb_smooth')

    print('working')

    if not os.path.exists(mdtb_dir):
        os.makedirs(mdtb_dir)
    
    if not os.path.exists(mdtb_smooth_dir):
        os.makedirs(mdtb_smooth_dir)

    # navigate to connectivity weight dir and grab files
    os.chdir(connect_dir)
    files = glob.glob('*.h5')
    print(files)

    # get remapping of file names
    data_dict = _remap()
    
    # copy best weights from connectivity to learning dir (change filenames)
    for file in files:
        for k,v in data_dict.items():
            if k in file:
                fname = file.replace(k, v).replace('ridge', 'RIDGE')

                src = os.path.join(connect_dir, file)
                dest = os.path.join(mdtb_dir, Path(fname).stem + '_mdtb.h5')
                copyfile(src, dest)

                # transpose the data first
                data = dd.io.load(dest)
                data['weights'] = data['weights'].T
                dd.io.save(dest, data)

                # smooth the cerebellar voxels and save to learning dir
                data['weights'] = np.nanmean(data['weights'], axis=1)
                data['weights'] = np.reshape(data['weights'], (len(data['weights']),1))
                dest_smooth = os.path.join(mdtb_smooth_dir, Path(fname).stem + '_mdtb_smooth.h5')
                dd.io.save(dest_smooth, data)

                print('worked')

if __name__ == "__main__":
    run()

