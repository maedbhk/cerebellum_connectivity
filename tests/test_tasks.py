from connectivity.tasks import get_betas, plot_task_maps_cerebellum

# get betas
Y, Y_info = get_betas(roi='cerebellum_suit', 
                    glm='glm7', 
                    exp='sc1',
                    averaging="sess", 
                    weighting=True)

# plot task betas
view = plot_task_maps_cerebellum(data=Y, data_info=Y_info, task='Instruct')

view