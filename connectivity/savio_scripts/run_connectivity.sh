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
#SBATCH --time=30:00:00
#
## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate
PYTHONPATH=$PYTHONPATH:/global/scratch/maedbhking/bin/
export PYTHONPATH

cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# atlases=(yeo7 yeo17 mdtb1002_007 mdtb1002_025 mdtb1002_050 mdtb1002_100 mdtb1002_150 mdtb1002_200)
# atlases=(tessels0042 tessels0162 tessels0362 tessels0642 tessels1002)
atlases=(Schaefer_7_100 Schaefer_7_200 Schaefer_7_300 fan gordon shen mdtb1002_300 mdtb1002_400 mdtb1002_400 mdtb1002_500)
models=(WTA ridge)

# train models
for ((m=0; m<${#models[@]}; m++)); do \
for ((a=0; a<${#atlases[@]}; a++)); do \
python3 script_mk.py --cortex=${atlases[a]} --model_type=${models[m]} --train_or_eval="train"; done; done

# evaluate models
python3 script_mk.py --train_or_eval="eval"