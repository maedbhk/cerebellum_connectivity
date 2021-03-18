# import packages
from pathlib import Path
import nibabel as nib
import numpy as np
from nilearn.image import mean_img
import os
import SUITPy.flatmap as flatmap
from nilearn.plotting import view_surf, plot_surf_roi
from nilearn.surface import load_surf_data

import connectivity.nib_utils as nio
from connectivity.data import convert_cerebellum_to_nifti

def nib_load(fpath):
    """ load nifti from disk

    Args: 
        fpath (str): full path to nifti img
    Returns: 
        returns nib obj
    """
    return nib.load(fpath)

def nib_save(img, fpath):
    """Save nifti to disk
    
    Args: 
        img (nib obj): 
        fpath (str): full path to nifti
    Returns: 
        Saves img to disk
    """
    nib.save(img, fpath)

def nib_mean(imgs):
    """ Get mean of nifti objs

    Args: 
        imgs (list): list of nib objs
    Returns: 
        mean nib obj
    """
    return mean_img(imgs)

def save_maps_cerebellum(data, fpath, group_average=True, gifti=True, nifti=True):
    """Takes list of np arrays, averages list and
    saves nifti and gifti map to disk

    Args: 
        data (np array): np array of shape (N x 6937)
        fpath (str): save path for output file
        group_average (bool): default is True, averages data np arrays 
        gifti (bool): default is True, saves gifti to fpath
        nifti (bool): default is True, saves nifti to fpath
    Returns: 
        saves nifti and/or gifti image to disk
    """
    # average data
    if group_average:
        data = np.nanmean(data, axis=0)

    # convert averaged cerebellum data array to nifti
    nib_obj = convert_cerebellum_to_nifti(data=data)[0]
    
    # save nifti to disk
    if nifti:
        nio.nib_save(img=nib_obj, fpath=fpath + '.nii') # this is temporary (to test bug in map)

    # map volume to surface
    surf_data = flatmap.vol_to_surf([fpath + '.nii'], space="SUIT")

    # make and save gifti image
    if gifti:
        gii_img = flatmap.make_func_gifti(data=surf_data, column_names=['col'])
        nio.nib_save(img=gii_img, fpath=fpath + '.gii')

def save_maps_cortex(data, fpath, atlas, group_average=True, hemisphere='R'):
    """Takes list of np arrays, averages list and
    saves gifti map to disk

    Args: 
        data (np array): np array of shape (N x 32492)
        fpath (str): save path for output file
        atlas (str): cortex atlas name (example: tesselsWB162)
        group_average (bool): default is True, averages data np arrays 
        hemisphere (str): 'R' or 'L'
    Returns: 
        saves gifti image to disk
    """
    # get anatomical structure
    if hemisphere=="R":
        anatomical_struct = 'CortexRight'
    elif hemisphere=="L":
        anatomical_struct = 'CortexLeft'
    
    # average data
    if group_average:
        data = np.nanmean(data, axis=0)
    
    # mesh surfaces
    dirs = const.Dirs()
    inflated_fpath = os.path.join(dirs.fs_lr_dir,f'fs_LR.32k.{hemisphere}.inflated.surf.gii')

    # get texture
    gii_path = os.path.join(dirs.reg_dir, 'data', 'group', f'{atlas}.{hemisphere}.label.gii')
    gii_data = nib.load(gii_path)
    texture = gii_data.darrays[0].data[:]

    # get start and end labels
    start_label = texture[texture!=0].min()
    end_label = texture[texture!=0].max()
    labels_all = np.arange(1, len(data)+1)

    # get relevant data
    data_hemi = data[start_label-1:end_label]
    labels = labels_all[start_label-1:end_label]

    # get texture
    texture = texture.astype(float)
    for vert in np.arange(len(texture)):
        label = texture[vert]
        if label != 0:
            texture[vert] = data_hemi[labels==label]

    # make gifti img
    func_gii = nio.make_func_gifti(data=texture.reshape(len(texture),1), anatomical_struct=anatomical_struct)
    nio.nib_save(img=func_gii, fpath=fpath)

def make_label_gifti(data, anatomical_struct='CortexLeft', label_names=None, column_names=None, label_RGBA=None):
    """
    Generates a label GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    Args:
        data (np array): shape (vertices x columns) 
        anatomical_struct (str): Anatomical Structure for the Meta-data default='CortexLeft'
        label_names (list or None): List of label names, default is None
        column_names (list or None): List of strings for column names, default is None
        label_RGBA (list or None): List of colors, default is None
    Returns:
        gifti (label GiftiImage)
    """
    num_verts, num_cols = data.shape
    num_labels = len(np.unique(data))

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_labels):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if label_RGBA is None:
        hsv = plt.cm.get_cmap('hsv', num_labels)
        color = hsv(np.linspace(0, 1, num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        label_RGBA = np.zeros([num_labels, 4])
        for i in range(num_labels):
            label_RGBA[i] = color[i]

    # Create label names
    if label_names is None:
        label_names = []
        for i in range(num_labels):
            label_names.append("label-{:02d}".format(i+1))

    # Create label.gii structure
    C = nib.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    E = nib.gifti.gifti.GiftiLabel()
    E.key = np.arange(label_names)
    E.label= label_names
    E.red = label_RGBA[:,0]
    E.green = label_RGBA[:,1]
    E.blue = label_RGBA[:,2]
    E.alpha = label_RGBA[:,3]

    D = list()
    for i in range(Q):
        d = nib.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_INT32',
            meta=nib.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nib.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.append(E)
    return gifti

def make_func_gifti(data, anatomical_struct='CortexLeft', column_names=None):
    """
    Generates a function GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    Args:
        data (np array): shape (vertices x columns) 
        anatomical_struct (str): Anatomical Structure for the Meta-data default='CortexLeft'
        column_names (list or None): List of strings for column names, default is None
    Returns:
        gifti (functional GiftiImage)
    """
    num_verts, num_cols = data.shape
  
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    C = nib.gifti.GiftiMetaData.from_dict({
    'AnatomicalStructurePrimary': anatomical_struct,
    'encoding': 'XML_BASE64_GZIP'})

    E = nib.gifti.gifti.GiftiLabel()
    E.key = 0
    E.label= '???'
    E.red = 1.0
    E.green = 1.0
    E.blue = 1.0
    E.alpha = 0.0

    D = list()
    for i in range(num_cols):
        d = nib.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_NONE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=nib.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    gifti = nib.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.append(E)

    return gifti

def view_cerebellum(data, cmap='jet', threshold=None, bg_map=None, cscale=None):
    """Visualize data on suit flatmap

    Args: 
        data (np array): np array of shape (28935 x 1)
        cmap (str): default is 'jet'
        threshold (int or None): default is None
        bg_map (str or None): default is None
        cscale (list or None): default is None
    """

    # full path to surface
    surf_dir = os.path.join(flatmap._surf_dir,'FLAT.surf.gii')

    # load topology
    # flatsurf = nib.load(surf_dir)
    # vertices = flatsurf.darrays[0].data
    # faces    = flatsurf.darrays[1].data

    # Determine underlay and assign color
    # underlay = nib.load(underlay)

    # Determine scale
    if cscale is None:
        cscale = [data.min(), data.max()]

    # nilearn seems to
    view = view_surf(surf_dir, data, bg_map=bg_map, cmap=cmap,
                        threshold=threshold, vmin=cscale[0], vmax=cscale[1])
    return view

def view_cortex(surf_mesh, surf_map, bg_map=None, map_type='func'):
    """Visualize data on inflated cortex

    Args: 
        surf_mesh (str or np array): *.inflated.surf.gii
        surf_map (str or np array): *.func.gii or *.label.gii
        bg_map (str or np array or None): 
        map_type (str): 'func' or 'label'
    """
    if map_type=="func":
        view = view_surf(surf_mesh=surf_mesh, 
                        surf_map=surf_map,
                        bg_map=bg_map,
                        )
                    
    elif map_type=="label":
        view = plot_surf_roi(surf_mesh=surf_mesh, 
                            roi_map=surf_map,
                            bg_map=bg_map,
                            )           
    return view
