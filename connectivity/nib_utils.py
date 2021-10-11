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
from SUITPy import flatmap
from nilearn.plotting import view_surf
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
            Anatomical Structure for the Meta-data default= 'CortexLeft' or 'L'; 'CortexRight' or 'R'
        label_names (list): 
            List of strings for label names
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors
    OUTPUTS:
        gifti (label GiftiImage)

    """

    if anatomical_struct=='L':
        anatomical_struct = 'CortexLeft'
    elif anatomical_struct=='R':
        anatomical_struct = 'CortexRight'

    try:
        num_verts, num_cols = data.shape
    except: 
        data = np.reshape(data, (len(data),1))
        num_verts, num_cols  = data.shape

    num_labels = len(np.unique(data))

    # check for 0 labels
    zero_label = 0 in data

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
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
        if zero_label:
            label_RGBA = np.vstack([[0,0,0,1], label_RGBA[1:,]])

    # Create label names
    if label_names is None:
        idx = 0
        if not zero_label:
            idx = 1
        for i in range(num_labels):
            label_names.append("label-{:02d}".format(i + idx))

    # Create label.gii structure
    C = nib.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    E_all = []
    for (label,rgba,name) in zip(np.arange(num_labels),label_RGBA,label_names):
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
            datatype='NIFTI_TYPE_FLOAT32', # was NIFTI_TYPE_INT32
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

    if anatomical_struct=='L':
        anatomical_struct = 'CortexLeft'
    elif anatomical_struct=='R':
        anatomical_struct = 'CortexRight'

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

    # labels = img.labeltable.get_labels_as_dict().values()
    label_dict = img.labeltable.get_labels_as_dict()

    return list(label_dict.values())

def get_gifti_columns(fpath):
    """get column names from gifti

    Args: 
        fpath (str or nib obj): full path to atlas (*.label.gii) or nib obj
    Returns: 
        column_names (list): list of column names
    """
    if isinstance(fpath, str):
        img = nib.load(fpath)
    else:
        img = fpath

    column_names = []
    for col in img.darrays:
        col_name =  list(col.metadata.values())[0]
        column_names.append(col_name)

    return column_names

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
    title=True,
    new_figure=True,
    save=False,
    cmap='jet',
    labels=None
    ):
    """Visualize (optionally saves) data on suit flatmap, plots either *.func.gii or *.label.gii data

    Args: 
        gifti (str): full path to gifti image
        cscale (list or None): default is None
        colorbar (bool): default is False.
        title (bool): default is True
        new_figure (bool): default is True. If false, appends to current axis. 
        save (bool): default is False
        cmap (str or matplotlib colormap): default is 'jet'
        labels (list): list of labels for *.label.gii. default is None
    """
    # figure out if 3D or 4D
    img = nib.load(gifti)

    # determine overlay and get metadata
    # get column names
    if '.func.' in gifti:
        overlay_type = 'func'
    elif '.label.' in gifti:
        overlay_type = 'label'
        _, _, cmap = get_gifti_colors(img)
        labels = get_gifti_labels(img)

    for (data, col) in zip(img.darrays, get_gifti_columns(img)):

        view = flatmap.plot(data.data, 
        overlay_type=overlay_type, 
        cscale=cscale, 
        cmap=cmap, 
        label_names=labels, 
        colorbar=colorbar, 
        new_figure=new_figure
        )

        # print title
        fname = Path(gifti).name.split('.')[0]
        if title:
            view.set_title(f'{fname}-{col}')

        # save to disk
        if save:
            dirs = const.Dirs()
            outpath = os.path.join(dirs.figure, f'{fname}-{col}.png')
            plt.savefig(outpath, dpi=300, format='png', bbox_inches='tight', pad_inches=0)

        plt.show()

def view_cortex(
    gifti, 
    surf='inflated',
    title=True,
    save=False,
    cmap='jet',
    column=None
    ):
    """Visualize (optionally saves) data on inflated cortex, plots either *.func.gii or *.label.gii data

    Args: 
        gifti (str): fullpath to file: *.func.gii or *.label.gii
        surf_mesh (str or None): fullpath to surface mesh file *.inflated.surf.gii. If None, takes mesh from `FS_LR` Dir
        title (bool): default is True
        save (bool): 'default is False',
        cmap (str or matplotlib colormap): 'default is "jet"'
        column (int or None): if gifti has multiple columns, you can choose which column to plot (default plots all)
    """
    # initialise directories
    dirs = const.Dirs()

    # figure out if 3D or 4D
    img = nib.load(gifti)

    if '.R.' in gifti:
        hemisphere = 'R'
    elif '.L.' in gifti:
        hemisphere = 'L'
    
    # get average mesh
    surf_mesh = os.path.join(dirs.reg_dir, 'data', 'group', f'fs_LR.32k.{hemisphere}.{surf}.surf.gii')

    # print title
    fname = Path(gifti).name.split('.')[0]
    title_name = None

    data_all = img.darrays
    cols = get_gifti_columns(img)

    if column is not None:
        data_all = [data_all[column]]
        cols = [cols[column]]

    for (data, col) in zip(data_all, cols):

        if title:
            title_name = f'{fname}-{col}'
        
        # plot to surface
        if '.func.' in gifti:
            view = view_surf(surf_mesh, 
                            surf_map=np.nan_to_num(data.data),  # was np.nan_to_num(data.data)
                            cmap=cmap, 
                            symmetric_cmap=False,
                            # view=orientation, 
                            title=title_name
                            )
            
        elif '.label.' in gifti:
            _, _, cmap= get_gifti_colors(img, ignore_0=False)
            # labels = get_gifti_labels(img)
            view = view_surf(surf_mesh, 
                            surf_map=data.data, # np.nan_to_num(data.data)
                            cmap=cmap, 
                            # view=orientation, 
                            symmetric_cmap=False,
                            title=title_name,
                            vmin=np.nanmin(data.data),
                            vmax=1 + np.nanmax(data.data),
                            colorbar=True
                            )
        
        view.open_in_browser() 

        if save:
            outpath = os.path.join(dirs.figure, f'{fname}-{col}.png')
            plt.savefig(outpath, dpi=300, format='png', bbox_inches='tight', pad_inches=0)

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

    
