#!/bin/bash

# !!!THIS NEEDS TO BE FIXED, THE KWARGS DON'T UPDATE!!!!

. /Users/maedbhking/.local/share/virtualenvs/cerebellum_connectivity-DQHkR575/bin/activate
cd /Users/maedbhking/Documents/cerebellum_connectivity/connectivity/models
python3 run_connectivity.py eval_on='sc1', train_on='sc2'