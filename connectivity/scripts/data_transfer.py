import os

def from_savio():
    #transfer eval data from savio to local
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/eval_summary_mk.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
   
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/*.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")

    # transfer best models
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/ridge_tessels1002_alpha_8 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/lasso_tessels1002_alpha_-2 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/WTA_tessels1002 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")

def to_savio():
    pass