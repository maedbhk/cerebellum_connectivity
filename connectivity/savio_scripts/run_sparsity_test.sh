#!/bin/bash
# Job name:
#SBATCH --job-name=test_sparsity
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
PYTHONPATH=$PYTHONPATH:/global/scratch/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/maedbhking/projects/cerebellum_connectivity/tests

model_names=(NTakeAll_tessels0042_2_positive NTakeAll_tessels0042_3_positive NTakeAll_tessels0042_4_positive NTakeAll_tessels0042_5_positive NTakeAll_tessels0042_10_positive)

for ((i=0; i<${#model_names[@]}; i++)); do \
python3 test_sparsity.py --model_name=${model_names[i]} --cortex='tessels0042' --train_exp='sc1'; done
