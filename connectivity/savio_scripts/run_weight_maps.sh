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
#SBATCH --time=02:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate

PYTHONPATH=$PYTHONPATH:/global/scratch/users/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/users/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# run weight maps
atlases=(MDTB10)
weights=(positive absolute)
data_type=(label func)

for ((w=0; w<${#weights[@]}; w++)); do \ 
    for ((a=0; a<${#atlases[@]}; a++)); do \
        for ((b=0; b<${#data_type[@]}; b++)); do \
            python3 script_weight_maps.py --atlas=${atlases[a]} --weights=${weights[w]} --data_type=${data_type[b]}; done; done; done

# connect_dir=/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/best_weights
# learn_dir=/global/scratch/users/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/derivatives/conn_models/train

# python3 run_transfer_weights.py --connect_dir=${connect_dir} --learn_dir=${learn_dir}