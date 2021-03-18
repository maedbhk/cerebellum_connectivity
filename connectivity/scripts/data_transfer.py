import os
from connectivity import visualize_summary as summary

def from_savio():
    # transfer eval data from savio to local
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/eval/")

    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/train/train_summary.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/train/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/train/train_summary.csv /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/train/")

    for exp in range(2):
        # get best train model (based on train CV)
        best_model = summary.get_best_model(train_exp=f"sc{2-exp}")
        os.system(f"rsync -avrz --include='*/' --include='{best_model}/*' --include='train_summary.csv' --exclude='*' maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/")


def to_savio():
    pass