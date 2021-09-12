#!/bin/bash
# Job name:
#SBATCH --job-name=connect_train_evalute
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
#SBATCH --time=05:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate

PYTHONPATH=$PYTHONPATH:/global/scratch/users/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/users/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# run connectivity models (eval)
# python3 script_mk.py --train_or_eval="eval"

# run difference scripts
# python3 script_compare_models.py

# run atlas
# python3 script_atlas.py

# run connectivity maps
atlases=(MDTB_10Regions MDTB_10subregions Buckner_17Networks)
weights=positive
data_type=(label func)

for ((a=0; a<${#atlases[@]}; a++)); do \
    for ((b=0; b<${#data_type[@]}; b++)); do \
        python3 script_connect_maps.py --atlas=${atlases[a]} --weights=${weights} --data_type=${data_type[b]}; done; done
