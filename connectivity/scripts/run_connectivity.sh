#!/bin/bash

. /Users/maedbhking/.local/share/virtualenvs/cerebellum_connectivity-DQHkR575/bin/activate
cd /Users/maedbhking/Documents/cerebellum_connectivity/connectivity/models
python3 train_evaluate.py eval_on='sc1', train_on='sc2'