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
source $(pipenv --venv)/bin/activate
cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# run connectivity models
python3 script_mk.py --cortex="tesselsWB162" --model_type="ridge"
python3 script_mk.py --cortex="tesselsWB362" --model_type="ridge"
python3 script_mk.py --cortex="tesselsWB642" --model_type="ridge"