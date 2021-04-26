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
#SBATCH --time=50:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate
PYTHONPATH=$PYTHONPATH:/global/scratch/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# run connectivity models
python3 script_mk.py --cortex="tessels0042" --model_type="NTakeAll" --train_or_eval="train" --positive=False
python3 script_mk.py --cortex="tessels0162" --model_type="NTakeAll" --train_or_eval="train" --positive=False
python3 script_mk.py --cortex="tessels0362" --model_type="NTakeAll" --train_or_eval="train" --positive=False
python3 script_mk.py --cortex="tessels0642" --model_type="NTakeAll" --train_or_eval="train" --positive=False
python3 script_mk.py --cortex="tessels1002" --model_type="NTakeAll" --train_or_eval="train" --positive=False

python3 script_mk.py --cortex="yeo7" --model_type="NTakeAll" --train_or_eval="train" --hyperparameter=[1] --positive=False
python3 script_mk.py --cortex="yeo17" --model_type="NTakeAll" --train_or_eval="train" --hyperparameter=[1] --positive=False