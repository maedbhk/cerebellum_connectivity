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
#SBATCH --time=60:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate
PYTHONPATH=$PYTHONPATH:/global/scratch/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

atlases=(tessels0042 tessels0162 tessels0362 tessels0642 tessels1002)
models=(ridge WTA)

# train models
for ((m=0; i<${#models[@]}; i++)); do \
for ((a=0; i<${#atlases[@]}; i++)); do \
python3 script_mk.py --cortex=${atlases[a]} --model_type=${models[m]} --train_or_eval="train"; done; done