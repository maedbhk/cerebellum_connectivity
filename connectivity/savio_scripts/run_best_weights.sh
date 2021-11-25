#!/bin/bash
# Job name:
#SBATCH --job-name=temp
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
#SBATCH --time=01:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate

PYTHONPATH=$PYTHONPATH:/global/scratch/users/maedbhking/bin/SUITPy/
export PYTHONPATH

cd /global/scratch/users/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# get best weights
methods=(lasso WTA) # ridge
for ((m=0; m<${#methods[@]}; m++)); do \
python3 script_best_weights.py --exp="sc1" --method=${methods[m]}; done

# transfer best weights
connect_dir=/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/best_weights
learn_dir=/global/scratch/users/maedbhking/projects/cerebellum_learning_connect/data/BIDS_dir/derivatives/conn_models/train

python3 run_transfer_weights.py --connect_dir=${connect_dir} --learn_dir=${learn_dir}