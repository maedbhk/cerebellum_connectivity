from connectivity.tasks import get_betas, plot_task_maps_cerebellum
import SUITPy as suit
import numpy as np
import connectivity.data as data



# get betas
"""Y, Y_info = get_betas(roi='cerebellum_suit', 
                    glm='glm7', 
                    exp='sc1',
                    averaging="sess", 
                    weighting=True)
"""
Ydata = data.Dataset(experiment="sc1", glm="glm7", roi="cerebellum_suit", subj_id="s02")
Ydata.load_mat()  # Import the data from Matlab
Y, Y_info = Ydata.get_data(averaging="exp")

# plot task betas: This is Maedbh's versopn: 
# view = plot_task_maps_cerebellum(data=Y, data_info=Y_info, task='Instruct')
# view
Yavr =  np.mean(Y[Y_info.inst==1,:],axis=0)
nii_func = data.convert_cerebellum_to_nifti(Yavr)
# Map to surface
map_func = suit.flatmap.vol_to_surf(nii_func)
# Plot the flatmap
suit.flatmap.plot(map_func)
