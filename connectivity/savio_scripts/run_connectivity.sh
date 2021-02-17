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
#SBATCH --time=1:00:00
#
## Command(s) to run:
module load python/3.7
. /global/home/users/maedbhking/.local/share/virtualenvs/cerebellum_connectivity-DQHkR575/bin/activate
cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/savio_scripts

# run connectivity models
python3 script_ridge_mk.py --model=train