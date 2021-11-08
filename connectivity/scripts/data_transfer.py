import os

def from_savio():
    #transfer eval data from savio to local
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/eval_summary_mk1.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
   
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/*.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")

def to_savio():
    pass