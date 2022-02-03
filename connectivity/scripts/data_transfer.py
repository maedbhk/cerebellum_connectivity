import os

def from_savio():
    # # transfer EVAL
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/ridge_tessels1002_alpha_8 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/lasso_tessels1002_alpha_-2 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/lasso_tessels0362_alpha_-3 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/WTA_tessels1002 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/WTA_tessels0042 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/WTA_yeo7 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")

    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/*.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
   
    # transfer TRAIN
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/*.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")
    
    # transfer CONVERGENCE MAPS
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/ridge_tessels1002_alpha_8/*.gii /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/lasso_tessels0362_alpha_-3/*.gii /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/ridge_tessels0042_alpha_4/*.gii /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/lasso_tessels0042_alpha_-3/*.gii /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/lasso_tessels1002_alpha_-2/*.gii /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/ridge_tessels0362_alpha_6/*.gii /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")

def to_savio():
    pass