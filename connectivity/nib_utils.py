# import packages
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from pathlib import Path
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
    numVerts, numCols = data.shape
    numLabels = len(np.unique(data))

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if len(column_names) == 0:
        for i in range(numLabels):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if len(label_RGBA) == 0:
        hsv = plt.cm.get_cmap('hsv',numLabels)
        color = hsv(np.linspace(0,1,numLabels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(numLabels)]
        label_RGBA = np.zeros([numLabels,4])
        for i in range(numLabels):
            label_RGBA[i] = color[i]

    # Create label names
    if len(label_names) == 0:
        for i in range(numLabels):
            label_names.append("label-{:02d}".format(i+1))

    # Create label.gii structure
    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})


    num_labels = np.arange(numLabels)
    E_all = []
    for (label,rgba,name) in zip(num_labels,label_RGBA,label_names):
        E = nb.gifti.gifti.GiftiLabel()
        E.key = label 
        E.label= name
        E.red = rgba[0]
        E.green = rgba[1]
        E.blue = rgba[2]
        E.alpha = rgba[3]
        E.rgba = rgba[:]
        E_all.append(E)

    D = list()
    for i in range(numCols):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL', 
            datatype='NIFTI_TYPE_INT32', # was NIFTI_TYPE_INT32
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
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

def view_cerebellum(data, cmap='CMRmap', threshold=None, bg_map=None, cscale=None, symmetric_cmap=False, title=None, vmin=None):
    """Visualize data on suit flatmap

    Args: 
        data (np array): np array of shape (28935 x 1) or str
        cmap (str): default is 'jet'
        threshold (int or None): default is None
        bg_map (str or None): default is None
        cscale (list or None): default is None
    """

    # full path to surface
    surf_mesh = os.path.join(flatmap._surf_dir,'FLAT.surf.gii')

    # load surf data from file
    if isinstance(data, str):
        fname = Path(data).name
        title = fname.split('.')[0]
        data = load_surf_data(data)

    # Determine underlay and assign color
    # underlay = nib.load(underlay)

    # convert 0 to nan (for plotting)
    data[data==0] = np.nan

    # Determine scale
    if cscale is None:
        cscale = [np.nanmin(data), np.nanmax(data)]
    
    if vmin is None:
        vmin = cscale[0]

    # nilearn seems to
    view = view_surf(surf_mesh, data, bg_map=bg_map, cmap=cmap,
                        threshold=threshold, vmin=vmin, vmax=cscale[1], 
                        symmetric_cmap=symmetric_cmap, title=title)
    # view = flatmap.plot(data, surf=surf_mesh, cscale=cscale)
    return view

def view_cortex(data, cmap='CMRmap', bg_map=None, cscale=None, hemisphere='R', atlas_type='inflated', symmetric_cmap=False, title=None, subset=None):
    """Visualize data on inflated cortex

    Args: 
        data (str or np array): *.func.gii or *.label.gii
        bg_map (str or np array or None): 
        map_type (str): 'func' or 'label'
        hemisphere (str): 'R' or 'L'
        atlas_type (str): 'inflated', 'very_inflated' (see fs_LR dir)
    """
    # initialise directories
    dirs = const.Dirs()

    # get surface mesh
    surf_mesh = os.path.join(dirs.fs_lr_dir, f'fs_LR.32k.{hemisphere}.{atlas_type}.surf.gii')

    # load surf data from file
    if isinstance(data, str):
        fname = Path(data).name
        title = fname.split('.')[0]
        data = load_surf_data(data)

    # subset data
    if subset:
        data = data[data==subset]
        
    # Determine scale
    if cscale is None:
        cscale = [np.nanmin(data), np.nanmax(data)]

    view = view_surf(surf_mesh=surf_mesh, 
                    surf_map=data,
                    bg_map=bg_map,
                    vmin=cscale[0], 
                    vmax=cscale[1],
                    cmap=cmap,
                    symmetric_cmap=symmetric_cmap,
                    title=title
                    )        
    return view


