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
from SUITPy import atlas as catlas
from nilearn.plotting import view_surf
# from surfplot import Plot

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

def get_gifti_anatomical_struct(
    fpath
    ):
    """
    Returns the primary anatomical structure for a gifti object (*.label.gii or *.func.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image 

    Returns:
        anatStruct (string):
            AnatomicalStructurePrimary attribute from gifti object
    """
    if isinstance(fpath, str):
        img = nib.load(fpath)
    else:
        img = fpath

    N = len(img._meta.data)
    anatStruct = []
    for i in range(N):
        if 'AnatomicalStructurePrimary' in img._meta.data[i].name:
            anatStruct.append(img._meta.data[i].value)
    return anatStruct

def get_random_rgba(gifti):

    img = nib.load(gifti)
    data = img.darrays[0].data
    label_RGBA=None
    
    num_labels = len(np.unique(data))
    zero_label = 0 in data
    
    hsv = plt.cm.get_cmap('hsv',num_labels)
    color = hsv(np.linspace(0,1,num_labels))
    # Shuffle the order so that colors are more visible
    color = color[np.random.permutation(num_labels)]
    label_RGBA = np.zeros([num_labels,4])
    for i in range(num_labels):
        label_RGBA[i] = color[i]
    if zero_label:
        label_RGBA = np.vstack([[0,0,0,1], label_RGBA[1:,]])
    
    cmap = LinearSegmentedColormap.from_list('mylist', label_RGBA, N=len(label_RGBA))
    
    return cmap

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

def get_cortical_atlases(
    atlas_keys=None, 
    hem='L'
    ):
    """returns: fpaths (list of str): list to all cortical atlases (*.label.gii) 
    Args:
        atlas_keys (None or list of str): default is None. 

    Returns: 
        fpaths (list of str): full path to cerebellar atlases
        atlases (list of str): names of cerebellar atlases
    """
    dirs = const.Dirs()

    fpaths = []
    fpath = os.path.join(dirs.reg_dir, 'data', 'group')
    for path in list(Path(fpath).rglob(f'*.{hem}.label.gii')):
        path = str(path)
        atlas = path.split('/')[-1].split('.')[0]
        if atlas_keys:
            if any(atlas_key in str(path) for atlas_key in atlas_keys):
                fpaths.append(path)
        else:
            fpaths.append(path)

    return fpaths

def get_cortical_surfaces(
    surf='flat', 
    hem='L'
    ):
    """Get cortical surfaces ('flat', 'inflated', 'pial' etc.)

    Args:
        surf (str): default is 'flat'
        hem (str): default is 'L'
    """

    dirs = const.Dirs()

    return os.path.join(dirs.reg_dir, 'data', 'group', f'fs_LR.32k.{hem}.{surf}.surf.gii')

def get_cerebellar_atlases(
    atlas_keys=None, 
    download_suit_atlases=False
    ):
    """returns: fpaths (list of str): list of full paths to cerebellar atlases

    Args:
        atlas_keys (None or list of str): default is None. 

    Returns: 
        fpaths (list of str): full path to cerebellar atlases
        atlases (list of str): names of cerebellar atlases
    """
    dirs = const.Dirs()

    if download_suit_atlases:
        catlas.fetch_king_2019(data_dir=dirs.cerebellar_atlases, data='atl')
        catlas.fetch_diedrichsen_2009(data_dir=dirs.cerebellar_atlases)
        catlas.fetch_buckner_2011(data_dir=dirs.cerebellar_atlases)
        catlas.fetch_xue_2021(data_dir=dirs.cerebellar_atlases)
        catlas.fetch_ji_2019(data_dir=dirs.cerebellar_atlases);
        
    fpaths = []
    # get atlases in cerebellar atlases
    fpath = os.path.join(dirs.base_dir, 'cerebellar_atlases')
    for path in list(Path(fpath).rglob('*.label.gii')):
        path = str(path)
        atlas = path.split('/')[-1].split('.')[0]
        if atlas_keys:
            if any(atlas_key in str(path) for atlas_key in atlas_keys):
                fpaths.append(path)
        else:
            fpaths.append(path)
    
    return fpaths

def view_cerebellum(
    gifti, 
    cscale=None, 
    colorbar=True, 
    title=True,
    new_figure=True,
    outpath=None,
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
        _, _, cmap = get_gifti_colors(img, ignore_0=False)
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
        if outpath is not None:
            plt.savefig(outpath, dpi=300, format='png', bbox_inches='tight', pad_inches=0)

        plt.show()

def view_cortex(
    gifti, 
    surf='inflated',
    hemisphere='L',
    data_type='func',
    title=None,
    outpath=None,
    # cmap='jet',
    column=None,
    colorbar=True
    ):
    """Visualize (optionally saves) data on inflated cortex, plots either *.func.gii or *.label.gii data

    Args: 
        gifti (str or nib gifti obj): fullpath to file: *.func.gii or *.label.gii
        surf (str): 'inflated', 'flat', 'pial', 'white'
        hemisphere (str): 'L' or 'R'. default is 'L'.
        data_type (str): 'func' or 'label'. default is 'func'. 
        title (str or None): default is None
        outpath (str or None): 'default is None, not saved to file
        cmap (str or matplotlib colormap): 'default is "jet"'
        column (int or None): if gifti has multiple columns, you can choose which column to plot (default plots all)
    """
    plt.clf()
    # initialise directories
    dirs = const.Dirs()

    # figure out if 3D or 4D
    if isinstance(gifti, str):
        img = nib.load(gifti)
    else:
        img = gifti

    if isinstance(gifti, str):
        if '.R.' in gifti:
            hemisphere = 'R'
        elif '.L.' in gifti:
            hemisphere = 'L'
        if '.func.' in gifti:
            data_type = 'func'
        elif '.label.' in gifti:
            data_type = 'label'
    
    # get average mesh
    surf_mesh = os.path.join(dirs.reg_dir, 'data', 'group', f'fs_LR.32k.{hemisphere}.{surf}.surf.gii')

    data_all = img.darrays
    cols = get_gifti_columns(img)

    if column is not None:
        data_all = [data_all[column]]
        cols = [cols[column]]

    for (data, col) in zip(data_all, cols):
        
        # plot to surface
        if data_type=='func':
            view = view_surf(surf_mesh, 
                            surf_map=np.nan_to_num(data.data),  # was np.nan_to_num(data.data)
                            # cmap=cmap, 
                            symmetric_cmap=False,
                            title=title
                            )
            
        elif data_type=='label':
            _, _, cmap= get_gifti_colors(img, ignore_0=False)
            # labels = get_gifti_labels(img)
            view = view_surf(surf_mesh, 
                            surf_map=data.data, # np.nan_to_num(data.data)
                            cmap=cmap, 
                            symmetric_cmap=False,
                            title=title,
                            vmin=np.nanmin(data.data),
                            vmax=1 + np.nanmax(data.data),
                            colorbar=colorbar
                            )
        
        view.open_in_browser() 

        # save to disk
        if outpath is not None:
            plt.savefig(outpath, dpi=300, format='png', bbox_inches='tight', pad_inches=0)

def view_cortex_inflated(
    giftis,
    colorbar=True, 
    borders=False,
    outpath=None,
    column=1
    ):
    """save cortical atlas to disk (and plot if plot=True)

    Args: 
        giftis (list of str or list of nib gifti obj): list has to be [left hemisphere, right hemisphere]. 
        surf_mesh (str): default is 'inflated'. other options: 'flat', 'pial'
        colorbar (bool): default is True
        borders (bool): default is False
        plot (bool): default is True
    """
    dirs = const.Dirs()

    # get surface mesh
    lh = get_cortical_surfaces(surf='inflated', hem='L')
    rh = get_cortical_surfaces(surf='inflated', hem='R')

    gifti_dict = {}
    for hem, gifti in zip(['L', 'R'], giftis):
        if isinstance(gifti, str):
            gifti_dict.update({hem: nib.load(gifti)})
        else:
            gifti_dict.update({hem: gifti})

    data_lh_all = gifti_dict['L'].darrays
    data_rh_all = gifti_dict['R'].darrays
    cols = get_gifti_columns(giftis[0])

    if column is not None:
        data_lh_all = [data_lh_all[column]]
        data_rh_all = [data_rh_all[column]]
        cols = [cols[column]]

    for (data_lh, data_rh, col) in zip(data_lh_all, data_rh_all, cols):

        p = Plot(lh, rh, size=(400, 200), zoom=1.2, views='lateral') # views='lateral', zoom=1.2, 

        p.add_layer({'left': np.nan_to_num(data_lh.data), 'right': np.nan_to_num(data_rh.data)},  cbar_label=col, as_outline=borders, cbar=colorbar) # cmap='YlOrBr_r',

        kws = {'location': 'right', 'label_direction': 45, 'decimals': 3,
       'fontsize': 16, 'n_ticks': 2, 'shrink': .15, 'aspect': 8,
       'draw_border': False}
        fig = p.build(cbar_kws=kws)

        plt.show()
        
        if outpath is not None:
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
    
    return fig

def view_atlas_cortex(
    atlas='yeo7',
    surf_mesh='inflated',  
    colorbar=True, 
    borders=False,
    ):
    """save cortical atlas to disk (and plot if plot=True)

    Args: 
        surf_mesh (str): default is 'inflated'. other options: 'flat', 'pial'
        atlas (str): default is 'yeo7'. 
        colorbar (bool): default is True
        borders (bool): default is False
        plot (bool): default is True
    """
    dirs = const.Dirs()

    # get surface mesh
    lh = get_cortical_surfaces(surf=surf_mesh, hem='L')
    rh = get_cortical_surfaces(surf=surf_mesh, hem='R')

    # get parcellation
    lh_data = get_cortical_atlases(atlas_keys=[atlas], hem='L')[0]
    rh_data = get_cortical_atlases(atlas_keys=[atlas], hem='R')[0]

    _, _, cmap = get_gifti_colors(fpath=lh_data)
    
    p = Plot(lh, rh)
    
    p.add_layer({'left': lh_data, 'right': rh_data}, cmap=cmap, cbar_label='Cortical Networks', as_outline=borders, cbar=colorbar) # 
    fig = p.build()
    plt.show()
    
    fig.savefig(os.path.join(dirs.figure, f'{atlas}-cortex.png'), dpi=300, bbox_inches='tight')
    
    return fig

def view_atlas_cerebellum(
    atlas='MDTB10_dseg', 
    colorbar=True,
    outpath=None,
    new_figure=True,
    labels=None
    ):
    """General purpose function for plotting (optionally saving) cerebellar atlas
    Args: 
        atlas (str): default is 'MDTB10_dseg'. other options: 'MDTB10-subregions', 'Buckner7', 'Buckner17', 
        structure (str): default is 'cerebellum'. other options: 'cortex'
        colorbar (bool): default is False. If False, saves colorbar separately to disk.
        outpath (str or None): outpath to file. if None, not saved to disk.
        new_figure (bool): default is True
        lbaels (list of int or None): default is None. 
    Returns:
        viewing object to visualize parcellations
    """
    gifti = get_cerebellar_atlases(atlas_keys=[atlas])[0]
    # view = view_cerebellum(gifti=gifti, colorbar=colorbar, title=title, outpath=outpath) \\    # figure out if 3D or 4D
    img = nib.load(gifti)

    _, _, cmap = get_gifti_colors(img, ignore_0=False)
    label_names = get_gifti_labels(img)
    data = img.darrays[0].data

    if labels is not None:
        for idx, num in enumerate(data):
            if num not in labels: 
                data[idx] = 0

    view = flatmap.plot(data, 
        overlay_type='label', 
        cmap=cmap, 
        label_names=label_names, 
        colorbar=colorbar, 
        new_figure=new_figure
        )

    # save to disk
    if outpath is not None:
        plt.savefig(outpath, dpi=300, format='png', bbox_inches='tight', pad_inches=0)

    plt.show()
    
    return view

def view_colorbar(
    atlas='yeo7', 
    structure='cortex',
    outpath=None,
    labels=None,
    orientation='vertical'
    ):
    """Makes colorbar for *.label.gii file
        
    Args:
        fpath (str): full path to *.label.gii
        outpath (str or None): default is None. file not saved to disk.
    """

    if structure=='cerebellum':
        fpath = get_cerebellar_atlases(atlas_keys=[atlas])[0]
    elif structure=='cortex':
        fpath = get_cortical_atlases(atlas_keys=[atlas], hem='L')[0]

    rotation = 90
    if orientation is 'horizontal':
        rotation = 45

    plt.figure()
    fig, ax = plt.subplots(figsize=(1,10)) # figsize=(1, 10)

    rgba, cpal, cmap = get_gifti_colors(fpath)

    if labels is None:
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
    cb3.ax.tick_params(axis='y', which='major', labelsize=30, labelrotation=rotation)

    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)

    return cb3

    
