import os

def from_savio():
    # transfer eval data from savio to local
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc2/conn_models/glm7/eval/* /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/conn_models/glm7/eval/")
    os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/conn_models/glm7/eval/* /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/conn_models/glm7/eval/")

    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc1/encoding/glm7/grey_nan_nonZeroInd.json /Users/maedbhking/Documents/cerebellum_connectivity/data/sc1/encoding/glm7/")
    # os.system("rsync -avrz maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/sc2/encoding/glm7/grey_nan_nonZeroInd.json /Users/maedbhking/Documents/cerebellum_connectivity/data/sc2/encoding/glm7/")

def to_savio():
    # transfer bash script
    os.system("rsync -avrz /Users/maedbhking/Documents/cerebellum_connectivity/connectivity/scripts/run_savio_jobs.sh maedbhking@dtn.brc.berkeley.edu:/global/scratch/maedbhking/projects/cerebellum_connectivity/data/savio_jobs/")