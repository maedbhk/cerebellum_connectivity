import os
import glob
from collections import defaultdict
import numpy as np
from scipy.stats import mode
import nibabel as nib
from SUITPy import flatmap

import connectivity.constants as const
import connectivity.io as cio
from connectivity import data as cdata
from connectivity import weights as cweights
from connectivity import visualize as summary

def run(
    exp='sc1',
    metric='nanmean',
    method='lasso',
    roi='tessels1002'
    ):
    """calculate sparsity maps for cerebellum

    Each cerebellar voxel is tagged with average (`metric`) of `roi` distances of non-zero coefs from `method` models

    Args: 
        exp (str): 'sc1' or 'sc2'. default is 'sc1'
        metric (str): default is 'nanmean'
        method (str or None): default is 'lasso'. Other options: 'NNLS', 'L2regression'. If None, all methods are chosen.
        roi (str or None): default is 'tessels1002'. If None, all rois are chosen. 
    """
    # set directory
    dirs = const.Dirs(exp_name=exp)

    # get best model (for each method and parcellation)
    models, rois = summary.get_best_models(train_exp=exp, method=method, roi=roi)

    for (model, cortex) in zip(models, rois):

        # get trained subject models
        fpath = os.path.join(dirs.conn_train_dir, model)
        model_fnames = glob.glob(os.path.join(fpath, '*.h5'))

        dist_all = defaultdict(list)
        for model in model_fnames:

            # read model data
            data = cio.read_hdf5(model)

            # calculate geometric mean of distances
            dist = cweights.sparsity_cortex(coef=data.coef_, roi=cortex, metric=metric)

            for k, v in dist.items():
                dist_all[k].append(v)

        # save maps to disk for cerebellum
        for k,v in dist_all.items():
            cweights.save_maps_cerebellum(data=np.stack(v, axis=0), 
                                fpath=os.path.join(fpath, f'group_{metric}_cerebellum_{k}'),
                                group='nanmean',
                                nifti=False)

if __name__ == "__main__":
    run()