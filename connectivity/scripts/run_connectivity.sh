#!/bin/bash

. /Users/maedbhking/.local/share/virtualenvs/cerebellum_connectivity-DQHkR575/bin/activate
cd /Users/maedbhking/Documents/cerebellum_connectivity/connectivity/models

python3 run_connectivity.py train=True evaluate=True train_on='"sc1"' eval_on='"sc2"' train_X_roi='"tesselsWB162"' eval_X_roi='"tesselsWB162"' lambdas='[0,10,20,50,100,200,500,1000]'