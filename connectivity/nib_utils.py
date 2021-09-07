# import packages
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import SUITPy.flatmap as flatmap
from nilearn.plotting import view_surf, plot_surf_roi
from nilearn.surface import load_surf_data

import connectivity.constants as const
from connectivity import data as cdata

def make_label_gifti_cortex(
    data, 
    anatomical_struct='CortexLeft', 
    label_names=None,
    column_names=None, 
    label_RGBA=None
    ):
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
    column_names=None
    ):
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

def get_gifti_colors(
    fpath,
    ignore_0=True
    ):
    """get gifti labels for fpath (should be *.label.gii)

    Args: 
        fpath (str or nib obj): full path to atlas
        ignore_0 (bool): default is True. ignores 0 index
    Returns: 
        rgba (np array): shape num_labels x num_rgba
        cpal (matplotlib color palette)
        cmap (matplotlib colormap)
    """
    dirs = const.Dirs()

    if isinstance(fpath, str):
        img = nib.load(fpath)
    else:
        img = fpath

    labels = img.labeltable.labels

    rgba = np.zeros((len(labels),4))
    for i,label in enumerate(labels):
        rgba[i,] = labels[i].rgba
    
    if ignore_0:
        rgba = rgba[1:]
        labels = labels[1:]

    cmap = LinearSegmentedColormap.from_list('mylist', rgba, N=len(rgba))
    mpl.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors=len(rgba))

    return rgba, cpal, cmap

def get_gifti_labels(
    fpath
    ):
    """get gifti labels for fpath (should be *.label.gii)

    Args: 
        fpath (str or nib obj): full path to atlas (*.label.gii) or nib obj
    Returns: 
        labels (list): list of label names
    """
    if isinstance(fpath, str):
        img = nib.load(fpath)
    else:
        img = fpath

    labels = img.labeltable.get_labels_as_dict().values()

    return list(labels)

def binarize_vol(
    imgs, 
    metric='max'
    ):
    """Binarizes niftis for `imgs` based on `metric`
    Args: 
        imgs (list of nib obj or list of str): list of nib objects or fullpath to niftis
        metric (str): 'max' or 'min'
    Returns: 
        nib obj
    """
    data_all = []
    for img in imgs:
        data_masked = cdata.read_suit_nii(img)
        data_all.append(data_masked)

    data = np.vstack(data_all)

    # binarize `data` based on max or min values
    if metric=='max':
        labels = np.argmax(data, axis=0)
    elif metric=='min':
        labels = np.argmin(data, axis=0)
    
    # compute 3D vol for `labels`
    nib_obj = cdata.convert_cerebellum_to_nifti(labels+1)

    return nib_obj[0]

def subtract_vol(
    imgs
    ):
    """Binarizes niftis for `imgs` based on `metric`
    Args: 
        imgs (list of nib obj or list of str): list of nib objects or fullpath to niftis
    Returns: 
        nib obj
    """

    if len(imgs)>2:
        print(Exception('there should be no more than two nib objs in `imgs`'))

    data_all = []
    for img in imgs:
        data_masked = cdata.read_suit_nii(img)
        data_all.append(data_masked)

    data = np.vstack(data_all)

    data_diff = data[1] - data[0]
    
    # compute 3D vol for `labels`
    nib_obj = cdata.convert_cerebellum_to_nifti(data_diff)

    return nib_obj[0]

def get_cortical_atlases(atlas_keys=None):
    """returns: fpaths (list of str): list to all cortical atlases (*.label.gii) 
    Args:
        atlas_keys (None or list of str): default is None. 

    Returns: 
        fpaths (list of str): full path to cerebellar atlases
        atlases (list of str): names of cerebellar atlases
    """
    dirs = const.Dirs()

    fpaths = []
    atlases = []
    fpath = os.path.join(dirs.reg_dir, 'data', 'group')
    for path in list(Path(fpath).rglob('*.label.gii')):
        path = str(path)
        atlas = path.split('/')[-1].split('.')[0]
        if atlas_keys:
            if any(atlas_key in str(path) for atlas_key in atlas_keys):
                fpaths.append(path)
                atlases.append(atlas)
        else:
            fpaths.append(path)
            atlases.append(atlas)

    return fpaths, atlases

def get_cerebellar_atlases(atlas_keys=None):
    """returns: fpaths (list of str): list of full paths to cerebellar atlases

    Args:
        atlas_keys (None or list of str): default is None. 

    Returns: 
        fpaths (list of str): full path to cerebellar atlases
        atlases (list of str): names of cerebellar atlases
    """
    dirs = const.Dirs()

    fpaths = []
    atlases = []
    # get atlases in cerebellar atlases
    fpath = os.path.join(dirs.base_dir, 'cerebellar_atlases')
    for path in list(Path(fpath).rglob('*.label.gii')):
        path = str(path)
        atlas = path.split('/')[-1].split('.')[0]
        if atlas_keys:
            if any(atlas_key in str(path) for atlas_key in atlas_keys):
                fpaths.append(path)
                atlases.append(atlas)
        else:
            fpaths.append(path)
            atlases.append(atlas)
    
    return fpaths, atlases

def view_cerebellum(
    gifti, 
    cscale=None, 
    colorbar=True, 
    title=None,
    new_figure=True,
    outpath=None
    ):
    """Visualize (optionally saves) data on suit flatmap, plots either *.func.gii or *.label.gii data

    Args: 
        gifti (str): full path to gifti image
        cscale (list or None): default is None
        colorbar (bool): default is False.
        title (bool): default is True
        new_figure (bool): default is True. If false, appends to current axis. 
        outpath (str or None): full path to filename. If None, figure is not saved to file
    """

    # full path to surface
    surf_mesh = os.path.join(flatmap._surf_dir,'FLAT.surf.gii')

    # determine overlay
    if '.func.' in gifti:
        overlay_type = 'func'
    elif '.label.' in gifti:
        overlay_type = 'label'

    view = flatmap.plot(gifti, surf=surf_mesh, overlay_type=overlay_type, cscale=cscale, colorbar=colorbar, new_figure=new_figure) # implement colorbar

    if title is not None:
        view.set_title(title)

    if outpath:
        if '.png' in outpath:
            format = 'png'
        elif '.svg' in outpath:
            format = 'svg'
        plt.savefig(outpath, dpi=300, format=format, bbox_inches='tight', pad_inches=0)

    return view

def view_cortex(
    gifti, 
    hemisphere='R', 
    cmap=None, 
    cscale=None, 
    atlas_type='inflated',  
    orientation='medial', 
    title=True,
    outpath=None
    ):
    """Visualize (optionally saves) data on inflated cortex, plots either *.func.gii or *.label.gii data

    Args: 
        gifti (str): fullpath to file: *.func.gii or *.label.gii
        hemisphere (str): 'R' or 'L'
        cmap (matplotlib colormap or None):
        cscale (int or None):
        atlas_type (str): 'inflated', 'very_inflated' (see fs_LR dir)
        orientation (str): 'medial' or 'lateral'
        title (bool): default is True
        outpath (str or None): default is None. file not saved to disk

    """
    # initialise directories
    dirs = const.Dirs()

    if '.R.' in gifti:
        hemisphere = 'R'
    elif '.L.' in gifti:
        hemisphere = 'L'

    # get surface mesh
    surf_mesh = os.path.join(dirs.reg_dir, 'data', 'group', f'fs_LR.32k.{hemisphere}.{atlas_type}.surf.gii')

    # Determine scale
    if ('.func.' in gifti and cscale is None):
        func_data = load_surf_data(gifti)
        cscale = [np.nanmin(func_data), np.nanmax(func_data)]

    fname = None
    if title:
        fname = Path(gifti).name
        fname.split('.')[0]
    
    if hemisphere=='L':
        orientation = 'lateral'

    if cmap is None:
        _, _, cmap = get_gifti_colors(fpath=gifti)

    view = plot_surf_roi(surf_mesh, gifti, cmap=cmap, view=orientation, title=fname) 

    if outpath:
        if '.png' in outpath:
            format = 'png'
        elif '.svg' in outpath:
            format = 'svg'
        plt.savefig(outpath, dpi=300, format=format, bbox_inches='tight', pad_inches=0)
    
    return view

def view_colorbar(
    fpath, 
    outpath=None
    ):
    """Makes colorbar for *.label.gii file
        
    Args:
        fpath (str): full path to *.label.gii
        outpath (str or None): default is None. file not saved to disk.
    """
    plt.figure()
    fig, ax = plt.subplots(figsize=(1,10)) # figsize=(1, 10)
    # fig, ax = plt.figure()

    rgba, cpal, cmap = get_gifti_colors(fpath)
    labels = get_gifti_labels(fpath)

    bounds = np.arange(cmap.N + 1)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap.reversed(cmap), 
                                    norm=norm,
                                    ticks=bounds,
                                    format='%s',
                                    orientation='vertical',
                                    )
    cb3.set_ticklabels(labels[::-1])  
    cb3.ax.tick_params(size=0)
    cb3.set_ticks(bounds+.5)
    cb3.ax.tick_params(axis='y', which='major', labelsize=30)

    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)

    return cb3

    
