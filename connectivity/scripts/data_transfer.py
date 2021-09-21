import os

def from_savio():
    #transfer eval data from savio to local
    # os.system("rsync -avrz --include='*/' --include='group_R_vox.func.gii' --include='group_R_vox.nii' --include='*eval_summary*' --exclude='*' maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    # os.system("rsync -avrz --include='*/' --include='group_R_vox.func.gii' --include='group_R_vox.nii' --include='*eval_summary*' --exclude='*' maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/eval/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/eval/")

    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/cerebellar_atlases/ /Users/maedbhking/Documents/cerebellum_connectivity/data/cerebellar_atlases/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc1/RegionOfInterest/data/group/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/RegionOfInterest/data/group/")
    
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/eval/ridge_tessels1002_alpha_8 /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/eval/")
    
    # for exp in range(2):
    #     os.system(f"rsync -avrz --include='*lasso*/' --include='*lasso*.gii' --include='*lasso*.nii' --exclude='*' maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/ /Users/maedbhking/Documents/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/")
        # os.system(f"rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/users/maedbhking/projects/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/*train_summary* /Users/maedbhking/Documents/cerebellum_connectivity/data/sc{2-exp}/conn_models/train/")

def to_savio():
    pass