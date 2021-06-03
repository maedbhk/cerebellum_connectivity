# import packages
import os
from pathlib import Path
import nibabel as nib
from nibabel.nifti1 import load
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn.image import mean_img
import SUITPy.flatmap as flatmap
from nilearn.plotting import view_surf, plot_surf_roi
from nilearn.surface import load_surf_data
import connectivity.constants as const

def make_label_gifti_cortex(data, anatomical_struct='CortexLeft', label_names=None, column_names=None, label_RGBA=None):
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
            label_names.append("label-{:02d}".format(i+1))

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

def make_func_gifti_cortex(data, anatomical_struct='CortexLeft', column_names=None):
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

def get_label_colors(fpath):
    """get rgba for atlas (given by fpath)

    Args: 
        fpath (str): full path to atlas
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

def view_cerebellum(data, threshold=None, cscale=None, symmetric_cmap=False, title=None):
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

    # nilearn seems to
    if viewer=='nilearn':
        view = view_surf(surf_mesh, data, cmap='CMRmap',
                        threshold=threshold, vmin=cscale[0], vmax=cscale[1], 
                        symmetric_cmap=symmetric_cmap)
    elif viewer=='suit':
        view = flatmap.plot(data, surf=surf_mesh, overlay_type=overlay_type, cscale=cscale)
    
    return view

def view_cortex(data, hemisphere='R', cmap=None, cscale=None, atlas_type='inflated', symmetric_cmap=False, title=None, view='medial'):
    """Visualize data on inflated cortex, plots either *.func.gii or *.label.gii data

    Args: 
        data (str): fullpath to file: *.func.gii or *.label.gii
        bg_map (str or np array or None): 
        map_type (str): 'func' or 'label'
        hemisphere (str): 'R' or 'L'
        atlas_type (str): 'inflated', 'very_inflated' (see fs_LR dir)
    """
    # initialise directories
    dirs = const.Dirs()

    # get surface mesh
    surf_mesh = os.path.join(dirs.reg_dir, 'data', 'group', f'fs_LR.32k.{hemisphere}.{atlas_type}.surf.gii')

    # load surf data from file
    fname = Path(data).name
    title = fname.split('.')[0]
        
    # Determine scale
    if ('.func.' in data and cscale is None):
        data = load_surf_data(data)
        cscale = [np.nanmin(data), np.nanmax(data)]

    if '.func.' in data:
        return view_surf(surf_mesh=surf_mesh, 
                        surf_map=data,
                        vmin=cscale[0], 
                        vmax=cscale[1],
                        cmap='CMRmap',
                        symmetric_cmap=symmetric_cmap,
                        # title=title
                        ) 
    elif '.label.' in data:   
        if hemisphere=='L':
            view = 'lateral'
        if cmap is None:
            _, cmap = get_label_colors(fpath=data)
        return plot_surf_roi(surf_mesh, data, cmap=cmap, view=view)    
    
