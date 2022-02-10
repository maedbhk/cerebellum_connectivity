#!/bin/bash
# Job name:
#SBATCH --job-name=make_atlas
#
# Account:
#SBATCH --account=fc_cerebellum
#
# Partition:
#SBATCH --partition=savio2

# Quality of Service:
#SBATCH --qos=savio_normal
#
# Wall clock limit:
#SBATCH --time=10:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate

PYTHONPATH=$PYTHONPATH:/global/scratch/users/maedbhking/bin/SUITPy/
export PYTHONPATH

cd /global/scratch/users/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

atlases=(MDTB10) # Buckner7 Buckner17 Anatom (problem with these atlases)
methods=(ridge lasso) # ridge

# # run cortical surface (voxels)
python3 script_surfaces.py --exp="sc1" --weights="nonzero" --method=${methods[m]} --regions="voxels"

# run cortical surfaces (rois)
for ((a=0; a<${#atlases[@]}; a++)); do \
for ((m=0; m<${#methods[@]}; m++)); do \
python3 script_surfaces.py --exp="sc1" --weights="nonzero" --method=${methods[m]} --regions="rois" --atlas=${atlases[a]}; done; done

# run dispersion (voxels)
for ((m=0; m<${#methods[@]}; m++)); do \
python3 script_dispersion.py --method=${methods[m]} --regions="voxels"; done

# run dispersion (rois)
for ((a=0; a<${#atlases[@]}; a++)); do \
for ((m=0; m<${#methods[@]}; m++)); do \
python3 script_dispersion.py --atlas=${atlases[a]} --method=${methods[m]} --regions="rois"; done; done

# # cortical weights
# for ((a=0; a<${#atlases[@]}; a++)); do \
# for ((m=0; m<${#methods[@]}; m++)); do \
# python3 script_cortical_weights.py --atlas=${atlases[a]} --method=${methods[m]} --exp="sc1"; done; done