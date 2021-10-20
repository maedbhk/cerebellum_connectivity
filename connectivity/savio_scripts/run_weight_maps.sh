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
python3 script_cortical_surface.py

# connect_dir=/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/best_weights
# learn_dir=/global/scratch/users/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/derivatives/conn_models/train

# python3 run_transfer_weights.py --connect_dir=${connect_dir} --learn_dir=${learn_dir}