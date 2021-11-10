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

## Command(s) to run:
module load python/3.7
source ~/.bash_profile
source $(pipenv --venv)/bin/activate

PYTHONPATH=$PYTHONPATH:/global/scratch/users/maedbhking/bin/SUITPy/
export PYTHONPATH

cd /global/scratch/users/maedbhking/projects/cerebellum_connectivity/connectivity/scripts

# atlases=(yeo7 yeo17 mdtb1002_007 mdtb1002_025 mdtb1002_050 mdtb1002_100 mdtb1002_150 mdtb1002_200)
# atlases=(tessels0042 tessels0162 tessels0362 tessels0642 tessels1002)
# atlases=(mdtb_wb_007 mdtb_wb_025 arslan_50 arslan_100 arslan_200 arslan_250)
atlases=(mdtb4002_wb_indv_7 mdtb4002_wb_indv_10 mdtb4002_wb_indv_17)
models=(WTA ridge)

# train models
for ((m=0; m<${#models[@]}; m++)); do \
for ((a=0; a<${#atlases[@]}; a++)); do \
python3 script_mk.py --cortex=${atlases[a]} --model_type=${models[m]} --train_or_eval="train"; done; done

# evaluate models
python3 script_mk.py --train_or_eval="eval"

# compare models
python3 script_compare_models.py

# run wta atlases
# atlases=(mdtb_wb_007 mdtb_wb_025)
# for ((a=0; a<${#atlases[@]}; a++)); do \
# python3 script_atlas.py --atlas=${atlases[a]}; done

