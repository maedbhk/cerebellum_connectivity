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
#SBATCH --time=40:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate
PYTHONPATH=$PYTHONPATH:/global/scratch/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# run connectivity models
# python3 script_mk.py --cortex="tesselsWB162" --model_type="ridge" --train_or_eval="train"
# python3 script_mk.py --cortex="tesselsWB362" --model_type="ridge" --train_or_eval="train"
# python3 script_mk.py --cortex="tesselsWB642" --model_type="ridge" --train_or_eval="train"

# python3 script_mk.py --cortex="tesselsWB162" --model_type="WTA" --train_or_eval="train"
# python3 script_mk.py --cortex="tesselsWB362" --model_type="WTA" --train_or_eval="train"
# python3 script_mk.py --cortex="tesselsWB642" --model_type="WTA" --train_or_eval="train"

python3 script_mk.py --cortex="tesselsWB162" --model_type="NNLS" --train_or_eval="train"
# python3 script_mk.py --cortex="tesselsWB362" --model_type="NNLS" --train_or_eval="train"
# python3 script_mk.py --cortex="tesselsWB642" --model_type="NNLS" --train_or_eval="train"


python3 script_mk.py --train_or_eval="eval"