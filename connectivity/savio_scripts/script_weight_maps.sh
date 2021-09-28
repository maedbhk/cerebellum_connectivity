#!/bin/bash
# Job name:
#SBATCH --job-name=lasso_maps
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

PYTHONPATH=$PYTHONPATH:/global/scratch/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# run lasso maps
atlases=(MDTB10)
weights=(positive absolute)
data_type=(label func)

for ((w=0; w<${#weights[@]}; w++)); do \ 
    for ((a=0; a<${#atlases[@]}; a++)); do \
        for ((b=0; b<${#data_type[@]}; b++)); do \
            python3 script_weight_maps.py --atlas=${atlases[a]} --weights=${weights[w]} --data_type=${data_type[b]}; done; done; done