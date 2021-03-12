from connectivity.data import Dataset
import connectivity.data as data
import connectivity.model as model
import connectivity.run as run
import numpy as np
import SUITPy as suit
from sklearn.linear_model import Ridge


def test_single_fit():
    # Get task by voxel data for cerebellum 
    Ydata = Dataset(glm="glm7", sn="s02", roi="cerebellum_suit")
    Ydata.load_mat()
    Y, T = Ydata.get_data(averaging="sess")
    # Get task by voxel data for cortex 
    Xdata = Dataset(glm=7, sn=2, roi="tesselsWB162")
    Xdata.load_mat()
    X, T = Xdata.get_data(averaging="sess")

    # Run the Ridge estimation model 
    R = model.L2regression(alpha=1)
    R.fit(X, Y)


def test_run_fit():
    config = run.get_default_train_config()
    config["name"] = "L2_WB162_A10"
    config["subjects"] = [2, 3, 4, 6]
    config["param"] = {"alpha": 10}
    config["weighting"] = 2
    config["train_exp"] = 1
    Model = run.train_models(config, save=True)
    pass


def test_run_eval():
    config = run.get_default_eval_config()
    config["name"] = "L2_WB162_A10"
    config["subjects"] = [3, 4, 6]
    config["eval_exp"] = 2
    config["weighting"] = 2
    T = run.eval_models(config)
    return T


def run_ridge():
    config = run.get_default_train_config()
    nameX = ["L2_WB162_Am2", "L2_WB162_A0", "L2_WB162_A2", "L2_WB162_A4", "L2_WB162_A6"]
    paramX = [
        {"alpha": np.exp(-2)},
        {"alpha": np.exp(0)},
        {"alpha": np.exp(2)},
        {"alpha": np.exp(4)},
        {"alpha": np.exp(6)},
    ]
    for i in range(len(nameX)):
        for e in range(2):
            config["name"] = nameX[i]
            config["param"] = paramX[i]
            config["weighting"] = 2
            config["train_exp"] = e + 1
            config["subjects"] = [3,4,6, 8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
            Model = run.train_models(config, save=True)
    pass


def test_dataset():
    Xdata = Dataset(experiment="sc1", glm="glm7", roi="cerebellum_suit", subj_id="s02")
    Xdata.load_mat()  # Import the data from Matlab
    T = Xdata.get_info_run()
    X, S = Xdata.get_data(averaging="exp")
    X1, S1 = Xdata.get_data(averaging="exp", subset=T.cond < 5)
    pass

def test_mapping_cerebellum():
    """
        Test the mapping to the cerebellar volume + surface + surface plotting
    """
    Xdata = Dataset(experiment="sc1", glm="glm7", roi="cerebellum_suit", subj_id="s02")
    Xdata.load_mat()  # Import the data from Matlab
    X, S = Xdata.get_data(averaging="exp")
    # Map to volume
    nii_func = data.convert_cerebellum_to_nifti(X[4,:])
    # Map to surface
    map_func = suit.flatmap.vol_to_surf(nii_func)
    # Plot the flatmap
    suit.flatmap.plot(map_func)
    pass

if __name__ == "__main__":
    test_mapping_cerebellum()
