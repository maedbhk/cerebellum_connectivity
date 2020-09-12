#!/bin/bash
# Job name:
#SBATCH --job-name=model_train_evaluate
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
#SBATCH --time=01:30:00
#
## Command(s) to run:
module load python/3.7
. /global/home/users/maedbhking/.local/share/virtualenvs/cerebellum_connectivity-DQHkR575/bin/activate
cd /global/scratch/maedbhking/projects/cerebellum_connectivity/connectivity/models/

python3 train_evaluate.py lambdas=[1,10,20,30,40,50,100,250,300,350,500,1000], train=False, train_on='sc1', eval_on='sc2', train_avg='sess', eval_avg='sess'
python3 train_evaluate.py lambdas=[1,10,20,30,40,50,100,250,300,350,500,1000], train_on='sc2', eval_on='sc1', train_avg='sess', eval_avg='sess'

python3 train_evaluate.py lambdas=[500], train=False, train_on='sc1', eval_on='sc2', train_avg='sess', eval_avg='sess', eval_save_maps=True
python3 train_evaluate.py lambdas=[500], train=False, train_on='sc2', eval_on='sc1', train_avg='sess', eval_avg='sess', eval_save_maps=True

