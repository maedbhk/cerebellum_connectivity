import os

def from_savio():
    # transfer eval data from savio to local
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/eval/")

    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/train/train_summary.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/train/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/train_summary.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")

def to_savio():
    pass