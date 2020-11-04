import nibabel
import os
import numpy as np
import tables

def save_table_file(filename, filedict):
    """Saves the variables in [filedict] in a hdf5 table file at [filename].
    """
    hf = tables.open_file(filename, mode="w", title="save_file")
    for vname, var in filedict.items():
        hf.create_array("/", vname, var)
    hf.close()
    
    
for subj in [2,3,4,6,8,9,10,12,14]:
    print('subject is:', subj)
    all_data = []
    for i in range(1, 17):
        data_dir = f'/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/imaging_data/s{subj:02}/rrun_{i:02}.nii'
        data = nibabel.load(data_dir).get_data().T
        all_data.append(data)
    print(f'all_data shape is :{np.array(all_data).shape}')
    
    data_dict = {'exp1': np.array(all_data)}
    save_fname = f'/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/imaging_data/s{subj:02}/rrun_sc1.hf5'
    save_table_file(save_fname, data_dict)
    all_data = []
    for i in range(17, 33):
        data_dir = f'/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/imaging_data/s{subj:02}/rrun_{i:02}.nii'
        data = nibabel.load(data_dir).get_data().T
        all_data.append(data)
    print(f'all_data shape is :{np.array(all_data).shape}')
    
    data_dict = {'exp2': np.array(all_data)}
    save_fname = f'/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/imaging_data/s{subj:02}/rrun_sc2.hf5'
    save_table_file(save_fname, data_dict)
    
    