# import packages
import os
from pathlib import Path
import nibabel as nib
from scipy.stats import mode
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn.input_data import NiftiMasker
import SUITPy.flatmap as flatmap
from nilearn.image import index_img, concat_imgs
from nilearn.plotting import view_surf, plot_surf_roi
from nilearn.surface import load_surf_data
import connectivity.constants as const

def make_label_gifti_cortex(
    data, 
    anatomical_struct='CortexLeft', 
    label_names=None, 
    column_names=None, 
    label_RGBA=None):
    """
    Generates a label GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        data (np.array):
             numVert x numCol data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'CortexLeft'
        label_names (list): 
            List of strings for label names
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors
    OUTPUTS:
        gifti (label GiftiImage)

    """
    try:
        num_verts, num_cols = data.shape
    except: 
        data = np.reshape(data, (len(data),1))
        num_verts, num_cols  = data.shape

    num_labels = len(np.unique(data))

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_labels):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if label_RGBA is None:
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        label_RGBA = np.zeros([num_labels,4])
        for i in range(num_labels):
            label_RGBA[i] = color[i]

    # Create label names
    if label_names is None:
        label_names = []
        for i in range(num_labels):
            label_names.append("label-{:02d}".format(i))

    # Create label.gii structure
    C = nib.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})


    num_labels = np.arange(num_labels)
    E_all = []
    for (label,rgba,name) in zip(num_labels,label_RGBA,label_names):
        E = nib.gifti.gifti.GiftiLabel()
        E.key = label 
        E.label= name
        E.red = rgba[0]
        E.green = rgba[1]
        E.blue = rgba[2]
        E.alpha = rgba[3]
        E.rgba = rgba[:]
        E_all.append(E)

    D = list()
    for i in range(num_cols):
        d = nib.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL', 
            datatype='NIFTI_TYPE_INT32', # was NIFTI_TYPE_INT32
            meta=nib.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nib.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
    return gifti

def make_func_gifti_cortex(
    data, 
    anatomical_struct='CortexLeft', 
    column_names=None):
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
    try:
        num_verts, num_cols = data.shape
    except: 
        data = np.reshape(data, (len(data),1))
        num_verts, num_cols  = data.shape
  
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

def make_gifti_cerebellum(
    data, 
    mask,
    outpath='/',
    stats='nanmean', 
    data_type='func',
    save_nifti=True, 
    save_gifti=True,
    column_names=[], 
    label_RGBA=[],
    label_names=[],
    ):
    """Takes data (np array) or 3D/4D nifti obj or str; optionally computes mean/mode along first dimension; optionally saves nifti and gifti map to disk

    If `data` is 3D/4D nifti obj or str, then nifti is masked using `mask` and np array (N x 6930) is returned

    Args: 
        data (np array or nib obj or str): np array of shape (N x 6930) or nib obj (3D or 4D) or str (fullpath to nifti)
        mask (nib obj or str): nib obj of mask or fullpath to mask
        outpath (str): save path for output file (must contain *.label.gii or *.func.gii)
        stats (str): 'nanmean', 'mode' or None if doing stats
        data_type (str): 'func' or 'label'
        save_nifti (bool): default is False, saves nifti to fpath
        column_names (list):
        label_RGBA (list):
        label_names (list):
    Returns: 
        saves gifti and/or nifti image to disk, returns gifti
    """
    if isinstance(mask, str):
        mask = nib.load(mask)
    elif isinstance(mask, nib.nifti1.Nifti1Image):
        pass

    if isinstance(data, str):
        data = nib.load(data)
        data = mask_vol(mask, data, output='2D')
    elif isinstance(data, nib.nifti1.Nifti1Image):
        data = mask_vol(mask, data, output='2D')
    elif isinstance(data, np.ndarray):
        pass

    # get mean or mode of data along first dim
    if stats=='nanmean':
        data = np.nanmean(data, axis=0)
    elif stats=='mode':
        data = mode(data, axis=0)
        data = data.mode[0]
    elif stats is None:
        pass

    # convert cerebellum data array to nifti
    imgs = mask_vol(mask, data, output='3D')
    
    # save nifti(s) to disk
    if save_nifti:
        fname = Path(outpath).name
        if len(imgs)>1:
            img = concat_imgs(imgs)
        else:
            img = imgs[0]
        nib.save(img, str(Path(outpath).parent) + '/' + fname.rsplit('.')[0] + '.nii')

    # make and save gifti
    if data_type=='label':
        surf_data = flatmap.vol_to_surf(imgs, space="SUIT", stats='mode')
        gii_img = flatmap.make_label_gifti(data=surf_data, label_names=label_names, column_names=column_names, label_RGBA=label_RGBA)
    elif data_type=='func':
        surf_data = flatmap.vol_to_surf(imgs, space="SUIT", stats='nanmean')
        gii_img = flatmap.make_func_gifti(data=surf_data, column_names=column_names)
    
    if save_gifti:
        nib.save(gii_img, outpath)
        print(f'saving gifti to {outpath}')
        return gii_img

def get_label_colors(fpath):
    """get rgba for atlas (given by fpath)

    Args: 
        fpath (str): full path to atlas (*.label.gii)
    Returns: 
        rgba (np array): shape num_labels x num_rgba
    """
    dirs = const.Dirs()

    img = nib.load(fpath)
    labels = img.labeltable.labels

    rgba = np.zeros((len(labels),4))
    for i,label in enumerate(labels):
        rgba[i,] = labels[i].rgba

    cmap = LinearSegmentedColormap.from_list('mylist', rgba)

    return rgba, cmap

def mask_vol(mask, data, output='2D'):
    """ mask volume using NiftiMasker, input can be volume (3D/4D Nifti1Image) or np array (n_cols x n_voxels) or str (to 3D/4D volume)

    If output is '3D' inverse transform is computed (go from 2D np array to list of 3D nifti(s))
    If output is '2D' then transform is computed (mask 3D nifti(s) and return 2D np array)

    Args: 
        mask (str or nib obj): should be `cerebellarGreySUIT3mm.nii`
        data (str or nib obj or np array): can be str to nifti (4D or 3D) or nifti obj (4D or 3D) or 2d array (n_cols x n_voxels)
        output (str): '2D' or '3D'. default is '2D'
    Returns: 
        np array shape (n_cols x n_voxels) if output='2D'
        list of nifti obj(s) if output='3D' (multiple niftis are returned if `data` is 4D nifti)
    """
    nifti_masker = NiftiMasker(standardize=False, mask_strategy='epi', memory_level=2,
                            smoothing_fwhm=0, memory="nilearn_cache") 

    # load mask if it's a string
    if isinstance(mask, str):
        mask = nib.load(mask)

    # fit the mask
    nifti_masker.fit(mask)

    # check vol format
    if isinstance(data, str):
        data = nib.load(data)
        fmri_masked = nifti_masker.transform(data) #  (n_time_points x n_voxels)
    elif isinstance(data, nib.nifti1.Nifti1Image):
        fmri_masked = nifti_masker.transform(data) #  (n_time_points x n_voxels)
    elif isinstance(data, np.ndarray): 
        try:
            num_vert, num_col = data.shape
        except: 
            data = np.reshape(data, (1,len(data)))
        fmri_masked = data

    # return masked data
    if output=="2D":
        return fmri_masked
    elif output=="3D":
        nib_obj = nifti_masker.inverse_transform(fmri_masked)
        nib_objs = []
        for i in np.arange(nib_obj.shape[3]):
            nib_objs.append(index_img(nib_obj,i))
        return nib_objs

def get_cortical_atlases():
    """returns: fpaths (list of str): list to all cortical atlases (*.label.gii) 
    """
    dirs = const.Dirs()

    fpaths = []
    fpath = os.path.join(dirs.reg_dir, 'data', 'group')
    for path in list(Path(fpath).rglob('*.label.gii')):
        # if any(atlas_key in str(path) for atlas_key in atlas_keys):
        fpaths.append(str(Path(path).name))

    return fpaths

def get_cerebellar_atlases():
    """returns: fpaths (list of str): list of full paths to cerebellar atlases
    """
    dirs = const.Dirs()

    fpaths = []
    # get atlases in cerebellar atlases
    fpath = os.path.join(dirs.base_dir, 'cerebellar_atlases')
    for path in list(Path(fpath).rglob('*.label.gii')):
        fpaths.append(str(Path(path).name))

    # get atlases in flatmap/surfaces
    for path in list(Path(flatmap._surf_dir).rglob('*.label.gii')):
        fpaths.extend([str(Path(path).name)])
    
    return fpaths

def binarize_vol(imgs, mask, metric='max'):
    """Binarizes niftis for `imgs` based on `metric`

    Args: 
        imgs (list of nib or list of str): list of nib objects or fullpath to niftis
        mask (nib or str): mask for `imgs`
        metric (str): 'max' or 'min'

    Returns: 
        nib obj
    """
    data_all = []
    for img in imgs:
        data_all.append(mask_vol(mask, data=img, output='2D'))

    data = np.vstack(data_all)

    # binarize `data` based on max or min values
    if metric=='max':
        labels = np.argmax(data, axis=0)
    elif metric=='min':
        labels = np.argmin(data, axis=0)
    
    # compute 3D vol for `labels`
    nib_obj = mask_vol(mask, data=labels+1, output='3D')

    return nib_obj

def view_cerebellum(
    data, 
    threshold=None, 
    cscale=None, 
    symmetric_cmap=False, 
    title=None,
    colorbar=True):
    """Visualize data on suit flatmap, plots either *.func.gii or *.label.gii data

    Args: 
        data (str): full path to gifti file
        cmap (str): default is 'jet'
        threshold (int or None): default is None
        bg_map (str or None): default is None
        cscale (list or None): default is None
    """

    # full path to surface
    surf_mesh = os.path.join(flatmap._surf_dir,'FLAT.surf.gii')

    # load surf data from file
    fname = Path(data).name
    title = fname.split('.')[0]

    if '.func.' in data:
        overlay_type = 'func'
        viewer = 'nilearn'
    elif '.label.' in data:
        overlay_type = 'label'
        viewer = 'suit'

    # Determine scale
    if ('.func.' in data and cscale is None):
        data = load_surf_data(data)
        cscale = [np.nanmin(data), np.nanmax(data)]

    # visualize
    # if viewer=='nilearn':
    #     return view_surf(surf_mesh, data, cmap='CMRmap',
    #                     threshold=threshold, vmin=cscale[0], vmax=cscale[1], 
    #                     symmetric_cmap=symmetric_cmap, colorbar=colorbar)
    # elif viewer=='suit':
    return flatmap.plot(data, surf=surf_mesh, overlay_type=overlay_type, cscale=cscale, colorbar=colorbar)

def view_cortex(
    data,  
    cmap=None, 
    cscale=None, 
    atlas_type='inflated', 
    symmetric_cmap=False, 
    title=None, 
    orientation='medial'):
    """Visualize data on inflated cortex, plots either *<hem>.func.gii or *<hem>.label.gii data

    Args: 
        data (str): fullpath to file: *<hem>.func.gii or *<hem>.label.gii
        bg_map (str or np array or None): 
        atlas_type (str): 'inflated', 'very_inflated' (see fs_LR dir)
    """
    # initialise directories
    dirs = const.Dirs()

    if '.R.' in data:
        hemisphere = 'R'
    elif '.L.' in data:
        hemisphere = 'L'
    else:
        hemisphere = 'R'

    # get surface mesh
    surf_mesh = os.path.join(dirs.reg_dir, 'data', 'group', f'fs_LR.32k.{hemisphere}.{atlas_type}.surf.gii')

    # load surf data from file
    fname = Path(data).name
    title = fname.split('.')[0]
        
    # Determine scale
    func_data = load_surf_data(data)
    if ('.func.' in data and cscale is None):
        cscale = [np.nanmin(func_data), np.nanmax(func_data)]

    if '.func.' in data:
        view = view_surf(surf_mesh=surf_mesh, 
                        surf_map=func_data,
                        vmin=cscale[0], 
                        vmax=cscale[1],
                        cmap='CMRmap',
                        symmetric_cmap=symmetric_cmap,
                        # title=title
                        ) 
    elif '.label.' in data:   
        if hemisphere=='L':
            orientation = 'lateral'
        if cmap is None:
            _, cmap = get_label_colors(fpath=data)
        view = plot_surf_roi(surf_mesh, data, cmap=cmap, view=orientation)    
    
    return view
    
