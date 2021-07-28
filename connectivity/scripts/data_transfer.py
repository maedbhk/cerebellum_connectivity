import os

def from_savio():
    #transfer eval data from savio to local
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/eval/")

    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/cerebellar_atlases/ /Users/maedbhking/Documents/cerebellum_connectivity/data/cerebellar_atlases/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/RegionOfInterest/data/group/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/RegionOfInterest/data/group/")
    
    for exp in range(2):
        os.system(f"rsync -avrz --include='*/' --include='*.gii' --include='*.nii' --include='*train_summary*' --exclude='*' maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/")

def to_savio():
    pass