import os
import glob
import click
from collections import defaultdict
import numpy as np
from scipy.stats import mode
import nibabel as nib
from pathlib import Path
import SUITPy.flatmap as flatmap

import connectivity.constants as const
import connectivity.io as cio
from connectivity import data as cdata
from connectivity import sparsity as csparse

def save_sparsity_maps(
    model_name, 
    cortex, 
    train_exp,
    metric
    ):
    """Save weight maps to disk for cortex and cerebellum

    Args: 
        model_name (str): model_name (folder in conn_train_dir)
        cortex (str): cortex model name (example: tesselsWB162)
        train_exp (str): 'sc1' or 'sc2'
        metric (str): 'nanmean', 'gmean', 'nanmedian'
    Returns: 
        saves nifti/gifti to disk
    """
    # set directory
    dirs = const.Dirs(exp_name=train_exp)

    # get model path
    fpath = os.path.join(dirs.conn_train_dir, model_name)

    # get trained subject models
    model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

    # get distance matrix
    distances = cdata.get_distance_matrix(roi=cortex)[0]

    dist_all = defaultdict(list)
    for model_fname in model_fnames:

        # read model data
        data = cio.read_hdf5(model_fname)

        # calculate geometric mean of distances
        # for NTakeAll tessels
        dist = csparse.geometric_distances(distances=distances, labels=data.labels_ntakeall, metric=metric)

        for k, v in dist.items():
            dist_all[k].append(v)

    # save maps to disk for cerebellum
    for k,v in dist_all.items():
        save_maps_cerebellum(data=np.stack(v, axis=0), 
                            fpath=os.path.join(fpath, f'group_ntakeall_cerebellum_{k}'),
                            group='nanmean',
                            nifti=False)

def save_maps_cerebellum(
    data, 
    fpath='/',
    group='nanmean', 
    gifti=True, 
    nifti=False, 
    column_names=[], 
    label_RGBA=[],
    label_names=[],
    ):
    """Takes data (np array), averages along first dimension
    saves nifti and gifti map to disk

    Args: 
        data (np array): np array of shape (N x 6937)
        fpath (str): save path for output file
        group (bool): default is 'nanmean' (for func data), other option is 'mode' (for label data) 
        gifti (bool): default is True, saves gifti to fpath
        nifti (bool): default is False, saves nifti to fpath
        column_names (list):
        label_RGBA (list):
        label_names (list):
    Returns: 
        saves nifti and/or gifti image to disk, returns gifti
    """
    num_cols, num_vox = data.shape

    # get mean or mode of data along first dim (first dim is usually subjects)
    if group=='nanmean':
        data = np.nanmean(data, axis=0)
    elif group=='mode':
        data = mode(data, axis=0)
        data = data.mode[0]
    else:
        print('need to group data by passing "nanmean" or "mode"')

    # convert averaged cerebellum data array to nifti
    nib_obj = cdata.convert_cerebellum_to_nifti(data=data)[0]
    
    # save nifti(s) to disk
    if nifti:
        nib.save(nib_obj, fpath + '.nii')

    # map volume to surface
    surf_data = flatmap.vol_to_surf([nib_obj], space="SUIT", stats=group)

    # make and save gifti image
    if group=='nanmean':
        gii_img = flatmap.make_func_gifti(data=surf_data, column_names=column_names)
        out_name = 'func'
    elif group=='mode':
        gii_img = flatmap.make_label_gifti(data=surf_data, label_names=label_names, column_names=column_names, label_RGBA=label_RGBA)
        out_name = 'label'
    if gifti:
        nib.save(gii_img, fpath + f'.{out_name}.gii')
    
    return gii_img

@click.command()
@click.option("--train_exp")
@click.option("--metric")

def run(train_exp='sc1',
        metric='nanmean'):

    rois = ['tessels0042', 'tessels0362', 'tessels0642', 'tessels1002']
    params = [2,3,4,5,10]

    for roi in rois:
        for param in params:
            save_sparsity_maps(model_name=f'NTakeAll_{roi}_{param}_positive', cortex=roi, train_exp='sc1', metric=metric)


if __name__ == "__main__":
    run()